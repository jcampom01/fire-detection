"""Move registered model versions through aliases: candidate -> @staging -> @production, with rollback.

Only reads metrics and moves aliases (no TensorFlow, no model download). Rules in promotion.json:
    minimums / maximums   absolute requirements on the run metrics (checked even with nothing in production)
    compare               the candidate must beat @production on this metric by at least min_improvement
Validation metrics are used to compare (the decision threshold fixes val recall at ~95%, so models differ in
false alarms); test metrics only as a sanity floor, so the choice is not tuned to the test set.

Commands:
    status      versions, their aliases, key metrics and last gate result; served version vs @production (drift)
    stage       run the gate on a version (default: latest) and, if it passes, move @staging to it
    deploy      re-run the gate on @staging, export it for TF Serving, move @production to it and serve it;
                the outgoing version gets @previous
    rollback    move @production back to @previous and serve it (already exported: only the config changes)

Deploying to TF Serving runs src/registry/export.py in a separate process, so this script never loads TensorFlow.
The export (slow, may fail) happens before any alias moves; activating the version is the last step.

Aliases (MLflow keeps no alias history, so every move also leaves tags on the versions):
    production / staging / previous, or production_<env> / staging_<env> / previous_<env> with --env.
    TF Serving here is the default environment: with --env only the aliases move (that site deploys on its own).

Usage:
    python src/registry/promote.py status
    python src/registry/promote.py stage --version 3 --dry-run
    python src/registry/promote.py deploy --env bcn
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")  # anonymous usage data sent to MLflow

from mlflow import MlflowClient  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402

RULES_PATH = Path(__file__).resolve().parent / "promotion.json"
EXPORT_SCRIPT = Path(__file__).resolve().parent / "export.py"
DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_SERVING_API_URL = "http://127.0.0.1:8080"


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class Registry:
    """Alias and tag operations on one registered model and one environment. With dry_run, writes are only printed."""

    def __init__(self, client, name, env, dry_run):
        self.client, self.name, self.dry_run = client, name, dry_run
        suffix = f"_{env}" if env else ""
        self.suffix = suffix
        self.production, self.staging, self.previous = (f"{a}{suffix}" for a in ("production", "staging", "previous"))

    # --- Reads ---

    def by_alias(self, alias):
        try:
            return self.client.get_model_version_by_alias(self.name, alias)
        except MlflowException:
            return None

    def versions(self):
        return sorted(self.client.search_model_versions(f"name='{self.name}'"), key=lambda v: int(v.version))

    def version(self, number):
        try:
            return self.client.get_model_version(self.name, str(number))
        except MlflowException:
            raise SystemExit(f"{self.name} has no version {number}") from None

    def run(self, version):
        return self.client.get_run(version.run_id)

    # --- Writes ---

    def _write(self, text, action):
        print(f"   {'[dry-run] would ' if self.dry_run else ''}{text}")
        if not self.dry_run:
            action()

    def set_alias(self, alias, version):
        self._write(f"set @{alias} -> v{version.version}",
                    lambda: self.client.set_registered_model_alias(self.name, alias, version.version))

    def delete_alias(self, alias):
        self._write(f"remove @{alias}", lambda: self.client.delete_registered_model_alias(self.name, alias))

    def tag(self, version, **tags):
        """Tag keys get the environment suffix, so the history of each environment stays separate."""
        for key, value in tags.items():
            key = f"{key}{self.suffix}"
            self._write(f"tag v{version.version} {key}={value}",
                        lambda k=key, v=value: self.client.set_model_version_tag(self.name, version.version, k, str(v)))


# --- Gate ---

def gate(reg, rules, candidate, production):
    """Return (passed, checks). Each check is (ok, text)."""
    run = reg.run(candidate)
    metrics, checks = run.data.metrics, []

    checks.append((run.info.status == "FINISHED", f"run status {run.info.status}"))
    checks.append((run.data.tags.get("debug") != "True", "not a debug run"))
    if production and production.version == candidate.version:
        checks.append((False, f"v{candidate.version} is already @{reg.production}"))

    for key, low in rules["minimums"].items():
        value = metrics.get(key)
        checks.append((value is not None and value >= low, f"{key} {value} >= {low}"))
    for key, high in rules["maximums"].items():
        value = metrics.get(key)
        checks.append((value is not None and value <= high, f"{key} {value} <= {high}"))

    if production and production.version != candidate.version:
        cmp = rules["compare"]
        key, sign = cmp["metric"], -1 if cmp["lower_is_better"] else 1
        cand, prod = metrics.get(key), reg.run(production).data.metrics.get(key)
        if cand is None or prod is None:
            checks.append((False, f"{key} missing (v{candidate.version}: {cand}, v{production.version}: {prod})"))
        else:
            improvement = sign * (cand - prod)
            checks.append((improvement >= cmp["min_improvement"],
                           f"{key} {cand:.4f} vs {prod:.4f} (v{production.version} @{reg.production}): "
                           f"improvement {improvement:+.4f}, needs >= {cmp['min_improvement']}"))
    else:
        checks.append((True, f"no @{reg.production} to compare with"))

    for ok, text in checks:
        print(f"   {'PASS' if ok else 'FAIL'}  {text}")
    return all(ok for ok, _ in checks), checks


def record_gate(reg, version, production, passed, checks):
    failed = [text for ok, text in checks if not ok]
    reg.tag(version, **{
        "gate.result": "passed" if passed else "failed",
        "gate.reason": "; ".join(failed) or "all checks passed",
        "gate.compared_to": f"v{production.version}" if production else "none",
        "gate.checked_at": now(),
    })


# --- Commands ---

def cmd_status(reg, rules, args):
    key = rules["compare"]["metric"]
    # search_model_versions does not fill version.aliases: read them from the registered model ({alias: version})
    aliases = {}
    for alias, number in reg.client.get_registered_model(reg.name).aliases.items():
        aliases.setdefault(str(number), []).append(alias)
    print(f"{reg.name}  (aliases: @{reg.production} @{reg.staging} @{reg.previous})\n")
    print(f"{'v':<4}{'run':<28}{key:>26}{'test.recall':>13}  {'gate':<8}aliases")
    for v in reg.versions():
        run = reg.run(v)
        m = run.data.metrics
        gate_result = v.tags.get(f"gate.result{reg.suffix}", "-")
        print(f"{v.version:<4}{run.info.run_name:<28}{m.get(key, float('nan')):>26.4f}"
              f"{m.get('test.recall', float('nan')):>13.4f}  {gate_result:<8}"
              f"{' '.join('@' + a for a in sorted(aliases.get(v.version, [])))}")

    if reg.suffix:
        return
    # Drift: does the service run what the registry says? (the alias could have been moved from the UI)
    production = reg.by_alias(reg.production)
    url = os.environ.get("SERVING_API_URL", DEFAULT_SERVING_API_URL)
    try:
        served = json.load(urllib.request.urlopen(f"{url}/model", timeout=5))["version"]
    except OSError as e:
        print(f"\nserved: unknown, the API at {url} does not answer ({e})")
        return
    expected = int(production.version) if production else None
    state = "OK" if served == expected else "DRIFT: run `deploy` or `rollback`, or export.py --version <v>"
    print(f"\n@{reg.production} = v{expected}   served = v{served}   {state}")


def cmd_stage(reg, rules, args):
    candidate = reg.version(args.version) if args.version else reg.versions()[-1]
    production = reg.by_alias(reg.production)
    print(f"gate for v{candidate.version} ({reg.run(candidate).info.run_name}):")
    passed, checks = gate(reg, rules, candidate, production)
    print(f"\n{'PASSED' if passed else 'FAILED'}:")
    record_gate(reg, candidate, production, passed, checks)
    if not passed:
        raise SystemExit(1)
    reg.set_alias(reg.staging, candidate)


def cmd_deploy(reg, rules, args):
    staging = reg.by_alias(reg.staging)
    if not staging:
        raise SystemExit(f"nothing in @{reg.staging}: run `stage` first")
    production = reg.by_alias(reg.production)

    # @production may have changed since staging: check again against the current one
    print(f"gate for v{staging.version} (@{reg.staging}), re-checked against the current @{reg.production}:")
    passed, checks = gate(reg, rules, staging, production)
    if not passed:
        print("\nFAILED:")
        record_gate(reg, staging, production, passed, checks)
        raise SystemExit(1)

    print("\nPASSED:")
    serve(reg, staging, activate=False)  # slow and may fail: before any alias moves
    stamp = now()
    if production:
        reg.set_alias(reg.previous, production)
        reg.tag(production, retired_at=stamp, replaced_by=f"v{staging.version}")
    reg.set_alias(reg.production, staging)
    reg.tag(staging, promoted_at=stamp, replaced=f"v{production.version}" if production else "none")
    reg.delete_alias(reg.staging)
    serve(reg, staging)


def cmd_rollback(reg, rules, args):
    previous = reg.by_alias(reg.previous)
    if not previous:
        raise SystemExit(f"nothing in @{reg.previous}: no deploy to undo")
    production = reg.by_alias(reg.production)

    print(f"rollback @{reg.production}: v{production.version if production else '-'} -> v{previous.version}")
    serve(reg, previous, activate=False)  # normally already exported: nothing to do
    stamp = now()
    if production:
        reg.tag(production, rolled_back_at=stamp)
    reg.set_alias(reg.production, previous)
    reg.tag(previous, restored_at=stamp)
    reg.delete_alias(reg.previous)
    serve(reg, previous)


def serve(reg, version, activate=True):
    """Run export.py for this version: export it (if needed) and, with activate, make TF Serving serve it."""
    if reg.suffix:
        if activate:
            print(f"   (--env {reg.suffix[1:]}: aliases only, TF Serving here is not that environment)")
        return
    cmd = [sys.executable, str(EXPORT_SCRIPT), "--version", str(version.version)]
    if not activate:
        cmd.append("--no-activate")
    print(f"   {'[dry-run] would run' if reg.dry_run else 'running'}: {' '.join(cmd[1:])}")
    if not reg.dry_run:
        subprocess.run(cmd, check=True)


def check_server(uri):
    if not uri.startswith("http"):
        return
    try:
        urllib.request.urlopen(f"{uri}/health", timeout=5)
    except OSError as e:
        raise SystemExit(f"MLflow not reachable at {uri} ({e}).\n"
                         f"Start it with: cd deploy/mlops && docker compose up -d") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "stage", "deploy", "rollback"])
    parser.add_argument("--version", type=int, help="version to stage (default: latest)")
    parser.add_argument("--env", default="", help="environment suffix for the aliases, e.g. bcn -> @production_bcn")
    parser.add_argument("--dry-run", action="store_true", help="show what would change without changing anything")
    args = parser.parse_args()

    rules = json.loads(RULES_PATH.read_text())
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    check_server(uri)
    reg = Registry(MlflowClient(uri), rules["registered_model"], args.env, args.dry_run)

    commands = {"status": cmd_status, "stage": cmd_stage, "deploy": cmd_deploy, "rollback": cmd_rollback}
    commands[args.command](reg, rules, args)


if __name__ == "__main__":
    main()

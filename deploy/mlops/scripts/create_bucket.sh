#!/usr/bin/env bash
# Create an S3 bucket in RustFS (idempotent). Run after `docker compose up -d`.
# Usage: ./scripts/create_bucket.sh [bucket]   (default: MLFLOW_BUCKET from .env)
set -euo pipefail

cd "$(dirname "$0")/.."
set -a; source .env; set +a

BUCKET="${1:-$MLFLOW_BUCKET}"
NETWORK="fire-mlops_default"

# --- aws-cli in a throwaway container, inside the compose network ---
export AWS_ACCESS_KEY_ID="$RUSTFS_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$RUSTFS_SECRET_KEY"

aws() {
  docker run --rm --network "$NETWORK" \
    -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
    -e AWS_DEFAULT_REGION=us-east-1 \
    -e AWS_ENDPOINT_URL=http://rustfs:9000 \
    amazon/aws-cli:2.39.0 "$@"
}

# --- Create the bucket only if it does not exist ---
if aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
  echo "Bucket '$BUCKET' already exists"
else
  aws s3 mb "s3://$BUCKET"
fi

aws s3 ls

#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

# Run pre-commit
echo "pre-commit run --all-files"
pre-commit run --all-files || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 1
fi
echo "Linting passed"
exit 0
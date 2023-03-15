#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

# Run black for automatic formatting
echo "black"
pre-commit run black || FAILURE=true

# Use flake8 to check for python code style violations, see .flake8 for details
echo "flake8"
pre-commit run flake8 || FAILURE=true

# Use mypy to check python types
echo "mypy"
pre-commit run mypy || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 1
fi
echo "Linting passed"
exit 0
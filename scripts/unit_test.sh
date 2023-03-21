#!/bin/bash
set -uo pipefail  # -u tells Bash to treat unset variables as errors, and -o pipefail tells Bash to return a failure status if any command in a pipeline fails.
set +e  # temporarily disables the "exit immediately on error" option (-e) in Bash, which allows the script to continue running even if some commands fail.

FAILURE=false

# pytest configuration set in pyproject.toml
echo "running pytest"
pytest || FAILURE=true # || operator means "or", so if pytest fails (i.e., returns a non-zero status code), then the FAILURE variable is set to true.

if [ "$FAILURE" = true ]; then
  echo "Unit tests failed"
  exit 1
fi
echo "Unit tests passed"
exit 0
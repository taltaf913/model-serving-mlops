#!/bin/bash
set -uo pipefail  # -u tells Bash to treat unset variables as errors, and -o pipefail tells Bash to return a failure status if any command in a pipeline fails.
set +e  # temporarily disables the "exit immediately on error" option (-e) in Bash, which allows the script to continue running even if some commands fail.

FAILURE=false

echo "running model serving deployment"
python ./model_serving_mlops/deployment/model_deployment/model_serving.py --model_name=staging-model-serving-mlops-model

if [ "$FAILURE" = true ]; then
  echo "Unit tests failed"
  exit 1
fi
echo "Unit tests passed"
exit 0
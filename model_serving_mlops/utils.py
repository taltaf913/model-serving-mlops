"""This module contains utils shared between different notebooks."""
import json
from pathlib import Path


def get_deployed_model_stage_for_env(env: str) -> str:
    """Get the model version stage.
    Stage under which the latest deployed model version can be found for the current environment.

    Args:
        env (str): Current environment

    Returns:
        str:  Model version stage
    """
    # For a registered model version to be served, it needs to be in either the Staging or Production
    # model registry stage
    # (https://learn.microsoft.com/azure/databricks/applications/machine-learning/manage-model-lifecycle/index#transition-a-model-stage).
    # For models in dev and staging, we deploy the model to the "Staging" stage, and in prod we deploy to the
    # "Production" stage
    _MODEL_STAGE_FOR_ENV = {
        "dev": "Staging",
        "staging": "Staging",
        "prod": "Production",
    }
    return _MODEL_STAGE_FOR_ENV[env]


def _get_ml_config_value(env: str, key: str):
    # Reading ml config from terraform output file for the respective key and env(staging/prod).

    conf_file_path = Path.cwd() / "model_serving_mlops" / "terraform" / "output" / f"{env}.json"

    try:
        with open(str(conf_file_path), "r") as handle:
            data = json.loads(handle.read())
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Unable to find file '{conf_file_path}'. Make sure ML config-as-code resources defined under "
            f"model-serving-mlops have been deployed to {env} (see model_serving_mlops"
            f"/terraform/README in the current git repo for details)"
        ) from e
    try:
        return data[key]["value"]
    except KeyError as e:
        raise RuntimeError(
            f"Unable to load key '{key}' for env '{env}'. Ensure that key '{key}' is defined " f"in f{conf_file_path}"
        ) from e


def _get_resource_name_suffix(test_mode: str):
    if test_mode:
        return "-test"
    else:
        return ""


def get_model_name(env: str, test_mode: bool = False):
    """Get the registered model name for the current environment.

    In dev or when running integration tests, we rely on a hardcoded model name.
    Otherwise, e.g. in production jobs, we read the model name from Terraform config-as-code output.

    Args:
        env (str): Current environment
        test_mode (bool, optional): Whether the notebook is running in test mode.. Defaults to False.

    Returns:
        _type_: Registered Model name.
    """
    if env == "dev" or test_mode:
        resource_name_suffix = _get_resource_name_suffix(test_mode)
        return f"model-serving-mlops-model{resource_name_suffix}"
    else:
        # Read ml model name from model_serving_mlops/terraform
        return _get_ml_config_value(env, "model-serving-mlops_model_name")

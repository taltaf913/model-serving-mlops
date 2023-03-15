"""This script creates a service principal in the staging and prod workspaces with appropriate permissions.
Output is writen to the specified json file.
"""
# !/usr/bin/env python

import json
import os
import pathlib
import subprocess
import sys


def run_cmd(cmd, **kwargs):
    current_script_dir = pathlib.Path(__file__).parent.resolve()
    return subprocess.run(cmd, check=True, cwd=current_script_dir, **kwargs)


def write_formatted_terraform_output(tf_output: dict, destination_file: str):
    """Write Terraform output to a destination filepath.
    Given a string containing JSON terraform output, i.e. a dict of string -> dict("value" -> string, "type" -> string,
    "sensitive" -> bool), extracts the "value" field from the dictionary and writes terraform JSON output
    to a destination file.

    Args:
        tf_output (dict): String containing JSON terraform output
        destination_file (str): Destination file path
    """
    tf_output_dict = json.loads(tf_output)
    secrets_dict = {key: tf_output_dict[key]["value"] for key in tf_output_dict}
    with open(destination_file, "w") as output_filename_handle:
        output_filename_handle.write(f"{json.dumps(secrets_dict, indent=2, sort_keys=True)}\n")


if __name__ == "__main__":
    run_cmd(["terraform", "init"])
    run_cmd(["terraform", "apply"] + sys.argv[1:])
    process = run_cmd(["terraform", "output", "-json"], capture_output=True)
    secrets_output_path = os.path.expanduser("~/.model-serving-mlops-cicd-service-principal-secrets.json")
    write_formatted_terraform_output(process.stdout, secrets_output_path)
    print(f"Wrote service principal secrets for CI/CD to {secrets_output_path}")

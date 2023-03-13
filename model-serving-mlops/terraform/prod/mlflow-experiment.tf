resource "databricks_mlflow_experiment" "experiment" {
  name        = "${local.mlflow_experiment_parent_dir}/${local.env_prefix}model-serving-mlops-experiment"
  description = "MLflow Experiment used to track runs for model-serving-mlops project."
}

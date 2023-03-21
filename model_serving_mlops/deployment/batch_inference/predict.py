import mlflow
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def predict_batch(
    spark_session: SparkSession,
    model_uri: str,
    input_table_name: str,
    output_table_name: str,
    model_version: str,
    ts: str,
) -> None:
    """Apply the model at the specified URI for batch inference.
    Applied on the table with name input_table_name, writing results to the table with name output_table_name.

    Args:
        spark_session (_type_): _description_
        model_uri (str): _description_
        input_table_name (str): Name of input table
        output_table_name (str): Name of output table
        model_version (int): MLflow Model Registry model version number
        ts (str): Timestamp
    """
    table = spark_session.table(input_table_name)

    predict = mlflow.pyfunc.spark_udf(spark_session, model_uri, result_type="string", env_manager="conda")
    output_df = (
        table.withColumn("prediction", predict(F.struct(*table.columns)))
        .withColumn("model_version", F.lit(model_version))
        .withColumn("inference_timestamp", F.to_timestamp(F.lit(ts)))
    )

    output_df.display()
    # Model predictions are written to the Delta table provided as input.
    # Delta is the default format in Databricks Runtime 8.0 and above.
    output_df.write.format("delta").mode("overwrite").saveAsTable(output_table_name)

#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

import os, warnings, sys, logging
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import mlflow.sklearn
from xgboost import XGBRegressor
from datetime import date
import cml.data_v1 as cmldata
import pyspark.pandas as ps
from sklearn.metrics import mean_squared_error



USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "TELCO_MLOPS_"+USERNAME
CONNECTION_NAME = "telefonicabr-az-dl"

DATE = date.today()
EXPERIMENT_NAME = "xgboostReg-autologging-{0}".format(USERNAME)

mlflow.set_experiment(EXPERIMENT_NAME)

conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

# ICEBERG SILVER TABLE
df = spark.sql("SELECT * FROM SPARK_CATALOG.TELCO_MEDALLION.PRODUCTS_SILVER")
df.count()
df.printSchema()

df_from_sql = ps.read_table('SPARK_CATALOG.TELCO_MEDALLION.PRODUCTS_SILVER')
df_from_sql = df_from_sql[["FL_VIVO_TOTAL", "TAXPIS", "TAXCOFINS", "TAXISS", "QTDD_BYTE_TFGD"]]
df = df_from_sql.to_pandas()
df = df.astype("float")

test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(df.drop("QTDD_BYTE_TFGD", axis=1), df["QTDD_BYTE_TFGD"], test_size=test_size)

import xgboost as xgb
from sklearn.model_selection import train_test_split
import mlflow


# Enable MLflow autologging
mlflow.xgboost.autolog()

with mlflow.start_run():
    # Train model
    n_splits=10
    random_state=1
    n_repeats=3
    max_depth=7

    model = xgb.XGBRegressor(objective='reg:squarederror', max_depth=max_depth)

    # define model evaluation method
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    # evaluate model
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

    # Evaluate and log metrics
    #test_pred = model.predict(X_test)
    mlflow.log_metric('rmse', mean_squared_error(y_test, test_pred, squared=False))

    mlflow.log_param("cv_splits", n_splits)
    mlflow.log_param("n_repeats", n_repeats)
    mlflow.log_param("cv_random_state", random_state)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("cv_mean_mae", scores.mean())
    mlflow.log_param("cv_stdv_mae", scores.std())

    model.fit(X_train, y_train)

    # Log model
    mlflow.xgboost.log_model(model, 'model')


def getLatestExperimentInfo(experimentName):
    """
    Method to capture the latest Experiment Id and Run ID for the provided experimentName
    """
    experimentId = mlflow.get_experiment_by_name(experimentName).experiment_id
    runsDf = mlflow.search_runs(experimentId, run_view_type=1)
    experimentId = runsDf.iloc[-1]['experiment_id']
    experimentRunId = runsDf.iloc[-1]['run_id']

    return experimentId, experimentRunId

experimentId, experimentRunId = getLatestExperimentInfo(EXPERIMENT_NAME)

#Replace Experiment Run ID here:
run = mlflow.get_run(experimentRunId)

pd.DataFrame(data=[run.data.params], index=["Value"]).T
pd.DataFrame(data=[run.data.metrics], index=["Value"]).T

client = mlflow.tracking.MlflowClient()
client.list_artifacts(run_id=run.info.run_id)

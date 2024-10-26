# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2025
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
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
#  Author: Paul de Fusco
#
# ###########################################################################

import os
import json
import mlflow
import pandas as pd
from prophet import Prophet, serialize
#import cdsw

# Load the model saved earlier.
prophetModel = '/home/cdsw/.experiments/lq9b-tbcu-23e9-64h9/b30r-w35u-y02z-lxlo/artifacts/prophet_model'
loaded_model = mlflow.prophet.load_model(prophetModel)

# *Note:* If you want to test this in a session, comment out the line
# `@cdsw.model_metrics` below. Don't forget to uncomment when you
# deploy, or it won't write the metrics to the database

#@cdsw.model_metrics
# This is the main function used for serving the model. It will take in the JSON formatted arguments , calculate the probablity of
# churn and create a LIME explainer explained instance and return that as JSON.

def predict(args):
    # Load JSON data
    #print("print arg types:")
    #print(type(args))
    #data = json.loads(args)
    #print("print type data:")
    #print(type(data))
    # Create DataFrame
    #df = pd.DataFrame({'ds': pd.to_datetime(data['dates'])})
    df = pd.DataFrame(data=args, dtype='datetime64[ns]', index=[1])
    df.columns=["ds"]
    df.head()
    # Predict on a Pandas DataFrame.
    forecast = loaded_model.predict(df)
    forecast = forecast[["ds", "yhat"]]

    return {"data": df.to_json(), "forecast": forecast.to_json()}


#args = """{"dates": [694224000000, 696902400000, 699408000000]}"""
#args = {"ds": "694224000000"}
#predict(args)

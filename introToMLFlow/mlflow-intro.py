#!/usr/bin/env python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Check localhost connection")

with mlflow.start_run():
    mlflow.log_metric("test",1)
    mlflow.log_metric("Maker",2)

with mlflow.start_run():
    mlflow.log_metric("test1",1)
    mlflow.log_metric("Maker1",2)

with mlflow.start_run():
    mlflow.log_metric("test2",1)
    mlflow.log_metric("Maker2",2)

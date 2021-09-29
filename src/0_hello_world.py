import os
import random
import mlflow
from dotenv import load_dotenv, find_dotenv

# load environment variables
load_dotenv()


if __name__ == '__main__':

    # log model params
    mlflow.log_param("alpha", 0.5)
    mlflow.log_param("beta", 0.5)

    for i in range(10):
        mlflow.log_metric("metric_1", random.random() + i, step=i)

    if not os.path.exists("models"):
        os.makedirs("models")
    with open("models/model.txt", "w") as f:
        f.write("hello world!")

    mlflow.log_artifact("models/model.txt")

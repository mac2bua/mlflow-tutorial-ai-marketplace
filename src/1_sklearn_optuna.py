"""
Source: https://github.com/optuna/optuna/blob/master/examples/mlflow/keras_mlflow.py

Optuna example that optimizes a neural network regressor for the
wine quality dataset using Keras and records hyperparameters and metrics using MLflow.
In this example, we optimize the learning rate and momentum of
stochastic gradient descent optimizer to minimize the validation mean squared error
for the wine quality regression.
"""

from keras.backend import clear_session
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import optuna
import pickle

from dotenv import load_dotenv
import numpy as np

# load environment variables
load_dotenv()

TEST_SIZE = 0.25
BATCHSIZE = 16
EPOCHS = 100
DEVELOPER_NAME = "ilia"

def standardize(data):
    return StandardScaler().fit_transform(data)


def create_model(num_features, trial):
    model = Sequential()
    model.add(
        Dense(
            num_features,
            activation="relu",
            kernel_initializer="normal",
            input_shape=(num_features,),
        )
    ),
    model.add(Dense(16, activation="relu", kernel_initializer="normal"))
    model.add(Dense(16, activation="relu", kernel_initializer="normal"))
    model.add(Dense(1, kernel_initializer="normal", activation="linear"))

    optimizer = SGD(
        lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        momentum=trial.suggest_float("momentum", 0.0, 1.0),
    )
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    return model


def mlflow_callback(study, trial):
    trial_value = trial.value if trial.value is not None else float("nan")
    with mlflow.start_run(run_name=study.study_name):
        mlflow.log_params(trial.params)
        mlflow.log_metrics({"rmse": np.sqrt(trial_value)})

def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    X, y = load_wine(return_X_y=True)
    X = standardize(X)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    model = create_model(X.shape[1], trial)
    model.fit(X_train, y_train, shuffle=True, batch_size=BATCHSIZE, epochs=EPOCHS, verbose=False)

    return model.evaluate(X_valid, y_valid, verbose=0)


if __name__ == "__main__":
    mlflow.set_experiment("optuna-example")

    study = optuna.create_study(study_name='keras-optuna')
    study.optimize(objective, n_trials=10, timeout=600, callbacks=[mlflow_callback])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
 
    mlflow.set_experiment("sklearn-example")
    with mlflow.start_run():
        mlflow.set_tag("developer", DEVELOPER_NAME)

        # Preprocess dataset
        X, y = load_wine(return_X_y=True)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42
        )

        # Log artifacts
        pickle.dump(scaler, open("scaler.pkl", "wb"))
        mlflow.log_artifact("scaler.pkl", artifact_path='preprocessing')

        # Log hyperparameters
        mlflow.log_params(trial.params)

        model = create_model(X.shape[1], trial)
        model.fit(
            X_train, y_train,   
            shuffle=True, 
            batch_size=BATCHSIZE, 
            epochs=EPOCHS, 
            verbose=False
        )

        print(model.evaluate(X_test, y_test, verbose=0))


        y_pred = model.predict(X_test)
        
        def eval_metrics(actual, pred):
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            return rmse, mae, r2
    
         # Evaluate trained model
        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        # Log metrics
        mlflow.log_metrics({
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })

        mlflow.keras.log_model(model, artifact_path='model')

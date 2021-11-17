from TaxiFareModel.data import get_data
from TaxiFareModel.data import clean_data
from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.utils import haversine_vectorized
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from google.cloud import storage
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import joblib
from TaxiFareModel.params import BUCKET_NAME, STORAGE_LOCATION

class Trainer():
    def __init__(self, X, y, experiment_name):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = experiment_name

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''

        pipe = Pipeline([
            ('linear_model',LassoCV(cv = 5 , n_alphas = 5) )
        ])
        self.pipeline = pipe
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri("https://mlflow.lewagon.co/")
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
        
    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(tr.set_pipeline(), 'model.joblib')
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')
        
        
    def extract_time_features(self, df):
        timezone_name = 'America/New_York'
        time_column = "pickup_datetime"
        df.index = pd.to_datetime(df[time_column])
        df.index = df.index.tz_convert(timezone_name)
        df["dow"] = df.index.weekday
        df["hour"] = df.index.hour
        df["month"] = df.index.month
        df["year"] = df.index.year
        return df.reset_index(drop=True)

    def extract_distance(self, df):
        df["distance"] = haversine_vectorized(df, 
                            start_lat="pickup_latitude", start_lon="pickup_longitude",
                            end_lat="dropoff_latitude", end_lon="dropoff_longitude")
        return df.reset_index(drop=True)




if __name__ == "__main__":
    #get the DataFrame
    df = get_data()
    print('Get data done...')
    #Clean the Data
    clean_data(df)
    print('Clean data data done...')
    #Add features from date
    
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    
    print('X and y defined...')
    # hold out
    
    print('X_train and y_train defined...')
    
    # train and build  pipeline
    tr = Trainer(X, y, "[FR][Paris][Danny C batch 722] TaciFareModel Savemodel")
    # Ajout des colonnes pour le time et la distance
    X = tr.extract_time_features(X)
    X = tr.extract_distance(X)
    # Suppression des colonnes non nécéssaire au calcul
    X = X[["distance", "hour", "dow", "passenger_count"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    tr = Trainer(X_train, y_train, "[FR][Paris][Danny C batch 722] TaciFareModel Savemodel")
    tr.set_pipeline()
    print('Pipeline set...')
    tr.run()
    print('Train data done...')
    # evaluate
    rmse = tr.evaluate(X_test, y_test)
    print('rmse calculated...')
    print(rmse)
    print('End')
    #mlflow
    tr.mlflow_log_metric("LinearRegression", rmse)
    tr.mlflow_log_param("train split", "0.1")
    tr.mlflow_log_param("model", "linear_model")
    tr.mlflow_log_param("student_name", "Danny")
    
    print("MLFlow Done")

    tr.save_model()
    
    print("Model Saved")
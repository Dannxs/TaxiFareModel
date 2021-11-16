from TaxiFareModel.data import get_data
from TaxiFareModel.data import clean_data
from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
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


if __name__ == "__main__":
    df = get_data()
    print('Get data done...')
    clean_data(df)
    print('Clean data data done...')
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    print('X and y defined...')
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    print('X_train and y_train defined...')
    
    # train and build  pipeline
    tr = Trainer(X_train, y_train)
    tr.set_pipeline()
    tr.run()
    print('Train data done...')
    # evaluate
    rmse = tr.evaluate(X_test, y_test)
    print('rmse calculated...')
    print(rmse)
    print('End')
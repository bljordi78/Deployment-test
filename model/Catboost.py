import warnings
warnings.filterwarnings("ignore") 
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import os

def modelCAT(df, target):
    """
    Function to train and save a catboost model.
    Takes as input a dataframe and the target variable.
    Saves a file with the model.
    Returns the model (that will be used for predictions)
    """   
    # Split into train/test
    X = df.drop(columns=[target])
    y = df[target]
 
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Initialize model
    model = CatBoostRegressor(verbose=0, random_state=123, cat_features=cat_features)

    # Fit on training set
    print(f"\nTraining CatBoost model...\n")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation metrics in original price scale
    r2 = round(r2_score(y_test, y_pred), 2)
    mae = round(mean_absolute_error(y_test, y_pred))
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)))

    print("CATBOOST MODEL METRICS ON TEST DATA:")
    print("R² Score:", r2)
    print("MAE:", mae)
    print("RMSE:", rmse)

    # Save the trained model
    os.makedirs("model", exist_ok=True)
    model_path = 'model/robocop_model.cbm'
    try:
        
        model.save_model(model_path)
        print(f"\nModel successfully saved to '{model_path}'\n")
    except Exception as e:
        print(f"Error saving model: {e}")

    return model


if __name__ == "__main__":
    df = pd.read_csv("model/data_cleaned.csv")
    model = modelCAT(df, target="price")
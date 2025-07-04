# Utils functions
from utils import regResults, plotCatBoostImportance

# Data wrangling libraries
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore") 
import os
os.makedirs("outputs", exist_ok=True)

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV, train_test_split

# Regression libraries
from catboost import CatBoostRegressor

# Output settings
pd.set_option('display.max_columns', None) # display all columns
pd.set_option('display.expand_frame_repr', False) # print all columns and in the same line
pd.set_option('display.max_colwidth', None) # display the full content of each cell
pd.set_option('display.float_format', lambda x: '%.3f' %x) # floats to be displayed with 3 decimal places


def modelCAT(df, target, obs, results_df=None):
    # Split into train/test
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Initialize model
    model = CatBoostRegressor(verbose=0, random_state=123)

    # Fit on training set
    model.fit(X_train, y_train)

    # Predict on train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Cross-validation on training set
    cv = KFold(n_splits=10, shuffle=True, random_state=123)
    y_cv_pred = cross_val_predict(model, X_train, y_train, cv=cv)

    # Get regression results
    train_results = regResults(obs, "CatBoost", y_train, y_train_pred)
    test_results = regResults(obs, "CatBoost", y_test, y_test_pred)
    cv_results = regResults(obs, "CatBoost CV", y_train, y_cv_pred)

    # Prepare results dictionary
    combined_result = {
        "Model": "CatBoost",
        "Features": obs,
        "Runtime (min)": None,
        **{k + " (Train)": v for k, v in train_results.items() if k not in ["Model", "Observations", "Features"]},
        **{k + " (CV)": v for k, v in cv_results.items() if k not in ["Model", "Observations", "Features"]},
        **{k + " (Test)": v for k, v in test_results.items() if k not in ["Model", "Observations", "Features"]},
    }

    # Append to results_df
    if results_df is not None:
        results_df = pd.concat([results_df, pd.DataFrame([combined_result])], ignore_index=True)
    else:
        results_df = pd.DataFrame([combined_result])

    # Feature importances from train set
    importances = model.get_feature_importance()
    feature_names = X_train.columns
    catboost_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    return results_df, catboost_importance

# Read the csv data
df = pd.read_csv("data/data_cleaned.csv")
target = 'price'
df_log = df.assign(price=np.log1p(df['price']))

# Rerunning our best model with all features
# and finding most important features
# and spliting the dataset with houses and apartments

df_hs = df_log[df_log['type_encoded'] == 1]  # Houses
df_apt = df_log[df_log['type_encoded'] == 0]  # Apartments

datasets = {
    "All properties": df_log,
    "Only Houses": df_hs,
    "Only Apartments": df_apt
}

results_df = None

for data, subset in datasets.items():

    start = time.time()
    results_df, catboost_importance = modelCAT(subset, target, data, results_df=results_df)
    end = time.time()

    code_run = round((end - start) / 60, 3)
    results_df.at[results_df.index[-1], "Runtime (min)"] = code_run

    results_df = results_df.sort_values("RMSE (Test)", ascending=True)

    print("\nSummary of Regression Results:")
    print(results_df)
    

    plotCatBoostImportance(catboost_importance, f"{data} - Feature Importances")

results_df.to_csv("outputs/results_catboost.csv", index=False)
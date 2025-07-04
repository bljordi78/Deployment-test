# Preprocessing libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Data wrangling libraries
import numpy as np

# Visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt


def regResults(features, model_name, y_true_log, y_pred_log):
    """
    Compute regression metrics (R², MAE, RMSE) on the original scale
    by converting log-transformed predictions and targets back to price.
    
    Parameters:
        features (str): Features used.
        model_name (str): Name of the model.
        y_true_log (array-like): Log-transformed true target values.
        y_pred_log (array-like): Log-transformed predicted values.
        
    Returns:
        dict: Regression metrics on the original price scale.
    """
    # Convert back from log1p to original scale
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "Features": features,
        "Model": model_name,
        "R² Score": r2,
        "MAE": mae,
        "RMSE": rmse
    }

def plotCatBoostImportance(catboost_importance, title):
    """
    Plot the features importance in descending order
    """
   
    plt.figure(figsize=(8, 5))
    sns.barplot(data=catboost_importance, x="Importance", y="Feature", color="steelblue")
    plt.title(f"CatBoost - {title}", fontsize=13, fontweight='bold')
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(f"outputs/{title}.png", dpi=300, bbox_inches='tight')

import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, root_mean_squared_error , r2_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import pickle
import optuna



def objective(trial):
    #memorizing point    
    # In Linear Regression, there's not much to tune,
    # but we can simulate feature selection or fit_intercept, normalize (deprecated in latest scikit-learn)
       fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
       model = LinearRegression(fit_intercept=fit_intercept)
       scores = cross_val_score(model, x_train, y_train, cv=3, scoring='neg_root_mean_squared_error')
       return scores.mean()  # higher is better (less negative)

    st.info("Tuning Linear Regression using Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    best_params = study.best_params
    st.success("Best parameters found using Optuna!")
    st.json(best_params)
    
    # UI for final model setup
    fit_intercept = st.selectbox("fit_intercept", [True, False], index=[True, False].index(best_params['fit_intercept']))
    
    if st.button("Train Final Model"):
        model_lin = LinearRegression(fit_intercept=fit_intercept)
        model_lin.fit(x_train, y_train)
        y_pred = model_lin.predict(x_test)
    
        st.subheader("Model Evaluation")
        st.write(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    
        # Save model to joblib
        joblib_buffer = io.BytesIO()
        joblib.dump(model_lin, joblib_buffer)
        st.download_button(
            label="Download Model (.joblib)",
            data=joblib_buffer.getvalue(),
            file_name="linear_regression_model.joblib",
            mime="application/octet-stream"
        )
    
        # Save model to pickle
        pickle_buffer = io.BytesIO()
        pickle.dump(model_lin, pickle_buffer)
        st.download_button(
            label="Download Model (.pkl)",
            data=pickle_buffer.getvalue(),
            file_name="linear_regression_model.pkl",
            mime="application/octet-stream"
        )
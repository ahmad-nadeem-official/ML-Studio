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
        params = {
            'C': trial.suggest_float('C', 1e-4, 10.0, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
        }

        # Handle solver + penalty compatibility
        if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
            raise optuna.exceptions.TrialPruned()
    
        model = LogisticRegression(**params, max_iter=1000, random_state=r_state)
        scores = cross_val_score(model, x_train, y_train, cv=3, scoring='accuracy')
        return np.mean(scores)
      
      st.spinner("Tuning Logistic Regression using Optuna...")
      study = optuna.create_study(direction="maximize")
      study.optimize(objective, n_trials=20)
      
      best_params = study.best_params
      st.success("Best hyperparameters found using Optuna!")
      st.json(best_params)
      
      # UI for final training
      col1, col2 = st.columns(2)
      with col1:
          C = st.number_input("C (Inverse of Regularization)", 0.0001, 10.0, best_params['C'], 0.01, format="%.4f")
      with col2:
          penalty = st.selectbox("Penalty", ['l1', 'l2'], index=['l1', 'l2'].index(best_params['penalty']))
      
      solver_options = ['liblinear', 'saga']
      compatible_solvers = [s for s in solver_options if not (penalty == 'l1' and s not in ['liblinear', 'saga'])]
      solver = st.selectbox("Solver", compatible_solvers, index=compatible_solvers.index(best_params['solver']))
      
      if st.button("Train Final Model"):
          model_log = LogisticRegression(
              C=C,
              penalty=penalty,
              solver=solver,
              max_iter=1000,
              random_state=r_state
          )
          model_log.fit(x_train, y_train)
          y_pred = model_log.predict(x_test)
      
          st.subheader("Model Evaluation")
          st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
          st.text("Classification Report:")
          st.text(classification_report(y_test, y_pred))
      
          # Save model to joblib
          joblib_buffer = io.BytesIO()
          joblib.dump(model_log, joblib_buffer)
          st.download_button(
              label="Download Model (.joblib)",
              data=joblib_buffer.getvalue(),
              file_name="logistic_regression_model.joblib",
              mime="application/octet-stream"
          )
      
          # Save model to pickle
          pickle_buffer = io.BytesIO()
          pickle.dump(model_log, pickle_buffer)
          st.download_button(
              label="Download Model (.pkl)",
              data=pickle_buffer.getvalue(),
              file_name="logistic_regression_model.pkl",
              mime="application/octet-stream"
          )    
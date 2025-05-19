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


 if model == "Random Forest (Classifier)":
      def objective(trial):
         params = {
             'n_estimators': trial.suggest_int('n_estimators', 50, 300),
             'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
             'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
         }

         model = RandomForestClassifier(**params, random_state=r_state)
         return cross_val_score(model, x_train, y_train, cv=3, scoring='accuracy').mean()

      st.info("Tuning Random Forest Classifier using Optuna...")
      study = optuna.create_study(direction="maximize")
      study.optimize(objective, n_trials=20)
      
      best_params = study.best_params
      st.success("Best hyperparameters found using Optuna!")
      st.json(best_params)
      
      # Add sliders/input for final training
      col1, col2, col3, col4 = st.columns(4)
      with col1:
          n_estimators = st.number_input("n_estimators", 10, 500, best_params['n_estimators'], 10)
      with col2:
          max_depth = st.number_input("max_depth", 1, 100, best_params['max_depth'], 1)
      with col3:
          min_samples_split = st.number_input("min_samples_split", 2, 20, best_params['min_samples_split'], 1)
      with col4:
          min_samples_leaf = st.number_input("min_samples_leaf", 1, 20, best_params['min_samples_leaf'], 1)
      
      if st.button("Train Final Model"):
          model_rfc = RandomForestClassifier(
              n_estimators=n_estimators,
              max_depth=max_depth,
              min_samples_split=min_samples_split,
              min_samples_leaf=min_samples_leaf,
              random_state=r_state
          )
          model_rfc.fit(x_train, y_train)
          y_pred = model_rfc.predict(x_test)
      
          st.subheader("Model Evaluation")
          st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
          st.text("Classification Report:")
          st.text(classification_report(y_test, y_pred))
          
          # Save model to joblib
          joblib_buffer = io.BytesIO()
          joblib.dump(model_rfc, joblib_buffer)
          st.download_button(
              label="Download Model (.joblib)",
              data=joblib_buffer.getvalue(),
              file_name="random_forest_model.joblib",
              mime="application/octet-stream"
          )
          
          # Save model to pickle
          pickle_buffer = io.BytesIO()
          pickle.dump(model_rfc, pickle_buffer)
          st.download_button(
              label="Download Model (.pkl)",
              data=pickle_buffer.getvalue(),
              file_name="random_forest_model.pkl",
              mime="application/octet-stream"
          ) 
  elif model == "Random Forest (Regressor)":
      def objective(trial):
         params = {
             'n_estimators': trial.suggest_int('n_estimators', 50, 300),
             'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
             'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
         }
         model = RandomForestRegressor(**params, random_state=r_state)
         scores = cross_val_score(model, x_train, y_train, cv=3, scoring='neg_root_mean_squared_error')
         return np.mean(scores)

      st.info("Tuning Random Forest Regressor using Optuna...")
      study = optuna.create_study(direction="maximize")
      study.optimize(objective, n_trials=20)
      
      best_params = study.best_params
      st.success("Best hyperparameters found using Optuna!")
      st.json(best_params)
      
      # UI to adjust hyperparameters
      col1, col2, col3, col4 = st.columns(4)
      with col1:
          n_estimators = st.number_input("n_estimators", 10, 500, best_params['n_estimators'], 10)
      with col2:
          max_depth = st.number_input("max_depth", 1, 100, best_params['max_depth'], 1)
      with col3:
          min_samples_split = st.number_input("min_samples_split", 2, 20, best_params['min_samples_split'], 1)
      with col4:
          min_samples_leaf = st.number_input("min_samples_leaf", 1, 20, best_params['min_samples_leaf'], 1)
      
      if st.button("Train Final Model"):
          model_rfr = RandomForestRegressor(
              n_estimators=n_estimators,
              max_depth=max_depth,
              min_samples_split=min_samples_split,
              min_samples_leaf=min_samples_leaf,
              random_state=r_state
          )
          model_rfr.fit(x_train, y_train)
          y_pred = model_rfr.predict(x_test)
      
          st.subheader("Model Evaluation")
          rmse = root_mean_squared_error(y_test, y_pred)
          r2 = r2_score(y_test, y_pred)
          acc = accuracy_score(y_test, y_pred)
          st.write(f"Accuracy score: {acc:.4f}")
          st.write(f"RMSE: {rmse:.4f}")
          st.write(f"R² Score: {r2:.4f}")
      
          # Save model to joblib
          joblib_buffer = io.BytesIO()
          joblib.dump(model_rfr, joblib_buffer)
          st.download_button(
              label="Download Model (.joblib)",
              data=joblib_buffer.getvalue(),
              file_name="random_forest_regressor.joblib",
              mime="application/octet-stream"
          )
      
          # Save model to pickle
          pickle_buffer = io.BytesIO()
          pickle.dump(model_rfr, pickle_buffer)
          st.download_button(
              label="Download Model (.pkl)",
              data=pickle_buffer.getvalue(),
              file_name="random_forest_regressor.pkl",
              mime="application/octet-stream"
          )
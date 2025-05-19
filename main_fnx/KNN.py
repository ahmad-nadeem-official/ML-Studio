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


  elif model == "KNN (Classifier)":
        def objective(trial):
          params = {
              'n_neighbors': trial.suggest_int('n_neighbors', 1, 30),
              'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
              'p': trial.suggest_int('p', 1, 2),  # 1: manhattan, 2: euclidean
          }
          model = KNeighborsClassifier(**params)
          return cross_val_score(model, x_train, y_train, cv=3, scoring='accuracy').mean()

        st.info("Tuning KNN Classifier using Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
    
        best_params = study.best_params
        st.success("Best hyperparameters found using Optuna!")
        st.json(best_params)
    
        # Final Training Parameters Input
        col1, col2, col3 = st.columns(3)
        with col1:
            n_neighbors = st.number_input("n_neighbors", min_value=1, max_value=30, value=best_params['n_neighbors'], step=1, key="knn_n_neighbors")
        with col2:
            weights = st.selectbox("weights", ['uniform', 'distance'], index=['uniform', 'distance'].index(best_params['weights']), key="knn_weights")
        with col3:
            p = st.number_input("p (1=manhattan, 2=euclidean)", min_value=1, max_value=2, value=best_params['p'], step=1, key="knn_p")
    
        if st.button("Train Final Model", key="knn_train_button"):
            # Use the current values from the widgets (which won't change unless user manually changes them)
            final_params = {
                'n_neighbors': n_neighbors,
                'weights': weights,
                'p': p
            }
            
            model_knn = KNeighborsClassifier(**final_params)
            model_knn.fit(x_train, y_train)
            y_pred = model_knn.predict(x_test)
    
            st.subheader("Model Evaluation")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            
            # Confusion Matrix
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            st.write(cm)
    
            # Save with joblib
            joblib_buffer = io.BytesIO()
            joblib.dump(model_knn, joblib_buffer)
            st.download_button(
                label="Download Model (.joblib)",
                data=joblib_buffer.getvalue(),
                file_name="knn_classifier.joblib",
                mime="application/octet-stream",
                key="knn_joblib_download"
            )
    
            # Save with pickle
            pickle_buffer = io.BytesIO()
            pickle.dump(model_knn, pickle_buffer)
            st.download_button(
                label="Download Model (.pkl)",
                data=pickle_buffer.getvalue(),
                file_name="knn_classifier.pkl",
                mime="application/octet-stream",
                key="knn_pickle_download"
            )

  elif model == "KNN (Regressor)":
        def objective(trial):
          params = {
              'n_neighbors': trial.suggest_int('n_neighbors', 1, 30),
              'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
              'p': trial.suggest_int('p', 1, 2),  # 1: manhattan, 2: euclidean
              'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
          }
          model = KNeighborsRegressor(**params)
          return cross_val_score(model, x_train, y_train, cv=3, scoring='r2').mean()

        st.info("Tuning KNN Regressor using Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
    
        best_params = study.best_params
        st.success("Best hyperparameters found using Optuna!")
        st.json(best_params)
    
        # Final Training Parameters Input
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_neighbors = st.number_input("n_neighbors", min_value=1, max_value=30, 
                                        value=best_params['n_neighbors'], step=1, 
                                        key="knn_reg_n_neighbors")
        with col2:
            weights = st.selectbox("weights", ['uniform', 'distance'], 
                                 index=['uniform', 'distance'].index(best_params['weights']), 
                                 key="knn_reg_weights")
        with col3:
            p = st.number_input("p (1=manhattan, 2=euclidean)", min_value=1, 
                              max_value=2, value=best_params['p'], step=1,
                              key="knn_reg_p")
        with col4:
            algorithm = st.selectbox("algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'],
                                   index=['auto', 'ball_tree', 'kd_tree', 'brute'].index(best_params['algorithm']),
                                   key="knn_reg_algorithm")
    
        if st.button("Train Final Model", key="knn_reg_train_button"):
            # Use the current widget values (won't change unless user modifies them)
            final_params = {
                'n_neighbors': n_neighbors,
                'weights': weights,
                'p': p,
                'algorithm': algorithm
            }
            
            model_knn = KNeighborsRegressor(**final_params)
            model_knn.fit(x_train, y_train)
            y_pred = model_knn.predict(x_test)
    
            st.subheader("Model Evaluation")
            st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
            st.write(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")
    
            # Save with joblib
            joblib_buffer = io.BytesIO()
            joblib.dump(model_knn, joblib_buffer)
            st.download_button(
                label="Download Model (.joblib)",
                data=joblib_buffer.getvalue(),
                file_name="knn_regressor.joblib",
                mime="application/octet-stream",
                key="knn_reg_joblib_download"
            )
    
            # Save with pickle
            pickle_buffer = io.BytesIO()
            pickle.dump(model_knn, pickle_buffer)
            st.download_button(
                label="Download Model (.pkl)",
                data=pickle_buffer.getvalue(),
                file_name="knn_regressor.pkl",
                mime="application/octet-stream",
                key="knn_reg_pickle_download"
            )

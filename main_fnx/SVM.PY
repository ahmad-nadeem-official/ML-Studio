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


   scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 0.1, 10, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']) else 'scale',
            'degree': trial.suggest_int('degree', 2, 5) if trial.suggest_categorical('kernel', ['poly']) else 3,
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
        model = SVC(**params, probability=True, random_state=r_state)
        return cross_val_score(model, x_train_scaled, y_train, cv=3, scoring='accuracy').mean()

    st.info("Tuning SVM Classifier using Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)  # More trials recommended for SVM

    best_params = study.best_params
    st.success("Best hyperparameters found using Optuna!")
    st.json(best_params)

    # Final Training Parameters Input
    col1, col2 = st.columns(2)
    with col1:
        kernel = st.selectbox("kernel", ['linear', 'poly', 'rbf', 'sigmoid'],
                            index=['linear', 'poly', 'rbf', 'sigmoid'].index(best_params['kernel']),
                            key="svc_kernel")
        C = st.number_input("C (Regularization)", min_value=0.1, max_value=10.0,
                          value=best_params['C'], step=0.1, key="svc_C")
        class_weight = st.selectbox("class_weight", [None, 'balanced'],
                                 index=[None, 'balanced'].index(best_params.get('class_weight', None)),
                                 key="svc_class_weight")
    with col2:
        gamma = st.selectbox("gamma", ['scale', 'auto'],
                           index=['scale', 'auto'].index(best_params.get('gamma', 'scale')),
                           key="svc_gamma") if best_params['kernel'] in ['rbf', 'poly', 'sigmoid'] else None
        degree = st.number_input("degree", min_value=2, max_value=5,
                              value=best_params.get('degree', 3), step=1,
                              key="svc_degree") if best_params['kernel'] == 'poly' else None

    if st.button("Train Final Model", key="svc_train_button"):
        # Prepare final parameters
        final_params = {
            'C': C,
            'kernel': kernel,
            'gamma': gamma if kernel in ['rbf', 'poly', 'sigmoid'] else 'scale',
            'degree': degree if kernel == 'poly' else 3,
            'class_weight': class_weight,
            'probability': True,
            'random_state': r_state
        }
        
        model_svc = SVC(**final_params)
        model_svc.fit(x_train_scaled, y_train)
        y_pred = model_svc.predict(x_test_scaled)
        y_proba = model_svc.predict_proba(x_test_scaled)

        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

        # Save complete package (model + scaler)
        model_package = {
            'model': model_svc,
            'scaler': scaler,
            'metadata': {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'parameters': final_params
            }
        }

        # Save with joblib
        joblib_buffer = io.BytesIO()
        joblib.dump(model_package, joblib_buffer)
        st.download_button(
            label="Download Model (.joblib)",
            data=joblib_buffer.getvalue(),
            file_name="svm_classifier.joblib",
            mime="application/octet-stream",
            key="svc_joblib_download"
        )

        # Save with pickle
        pickle_buffer = io.BytesIO()
        pickle.dump(model_package, pickle_buffer)
        st.download_button(
            label="Download Model (.pkl)",
            data=pickle_buffer.getvalue(),
            file_name="svm_classifier.pkl",
            mime="application/octet-stream",
            key="svc_pickle_download"
        )
  elif model == "SVM (Regressor)":
            # Scale data for better SVM performance
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    def objective(trial):
        params = {
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'C': trial.suggest_float('C', 0.1, 10, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']) else 'scale',
            'degree': trial.suggest_int('degree', 2, 5) if trial.suggest_categorical('kernel', ['poly']) else 3,
            'epsilon': trial.suggest_float('epsilon', 0.01, 0.2)
        }
        model = SVR(**params)
        return cross_val_score(model, x_train_scaled, y_train, cv=3, scoring='r2').mean()

    st.info("Tuning SVM Regressor using Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)  # More trials recommended for SVM

    best_params = study.best_params
    st.success("Best hyperparameters found using Optuna!")
    st.json(best_params)

    # Final Training Parameters Input
    col1, col2, col3 = st.columns(3)
    with col1:
        kernel = st.selectbox("kernel", ['linear', 'poly', 'rbf', 'sigmoid'], 
                            index=['linear', 'poly', 'rbf', 'sigmoid'].index(best_params['kernel']),
                            key="svr_kernel")
        C = st.number_input("C (Regularization)", min_value=0.1, max_value=10.0, 
                           value=best_params['C'], step=0.1, key="svr_C")
    with col2:
        gamma = st.selectbox("gamma", ['scale', 'auto'], 
                           index=['scale', 'auto'].index(best_params.get('gamma', 'scale')),
                           key="svr_gamma") if best_params['kernel'] in ['rbf', 'poly', 'sigmoid'] else None
        epsilon = st.number_input("epsilon", min_value=0.01, max_value=0.2, 
                                value=best_params['epsilon'], step=0.01, key="svr_epsilon")
    with col3:
        degree = st.number_input("degree", min_value=2, max_value=5, 
                               value=best_params.get('degree', 3), step=1,
                               key="svr_degree") if best_params['kernel'] == 'poly' else None

    if st.button("Train Final Model", key="svr_train_button"):
        # Prepare final parameters
        final_params = {
            'kernel': kernel,
            'C': C,
            'epsilon': epsilon,
            'gamma': gamma if kernel in ['rbf', 'poly', 'sigmoid'] else 'scale',
            'degree': degree if kernel == 'poly' else 3
        }
        
        model_svr = SVR(**final_params)
        model_svr.fit(x_train_scaled, y_train)
        y_pred = model_svr.predict(x_test_scaled)

        st.subheader("Model Evaluation")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
        st.write(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")

        # Save both model and scaler
        model_package = {
            'model': model_svr,
            'scaler': scaler,
            'metadata': {
                'r2_score': r2_score(y_test, y_pred),
                'rmse': root_mean_squared_error(y_test, y_pred),
                'parameters': final_params
            }
        }

        # Save with joblib
        joblib_buffer = io.BytesIO()
        joblib.dump(model_package, joblib_buffer)
        st.download_button(
            label="Download Model (.joblib)",
            data=joblib_buffer.getvalue(),
            file_name="svm_regressor.joblib",
            mime="application/octet-stream",
            key="svr_joblib_download"
        )

        # Save with pickle
        pickle_buffer = io.BytesIO()
        pickle.dump(model_package, pickle_buffer)
        st.download_button(
            label="Download Model (.pkl)",
            data=pickle_buffer.getvalue(),
            file_name="svm_regressor.pkl",
            mime="application/octet-stream",
            key="svr_pickle_download"
        )
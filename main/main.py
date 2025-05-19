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


# page configuration
st.set_page_config(page_title="ML Studio", page_icon="🤖", layout="wide")

st.title("ML Studio")
st.write(
    "This is a machine learning studio that allows you to upload a CSV file, visualize the data, and train a machine learning model."
)

#precautions
with st.sidebar.expander("⚠️ Important: Prepare Your Data Before Training"):
    st.markdown("""
### 📌 Before You Upload Your Dataset:

This ML Studio is designed **only for training machine learning models**. It does **not perform any automatic data cleaning**.

To get the best results:
- ✅ Make sure your CSV is **cleaned, preprocessed, and ready for modeling**.
- 🧼 Your data should have:
  - No missing values (or properly handled)
  - No irrelevant or duplicate columns
  - Balanced and properly encoded categorical features
  - Standardized or normalized numerical data

---

### 🚀 Need Help Cleaning Your Dataset?

If your CSV is not cleaned yet, no problem! Use my **Google Colab Data Cleaning Tool** where you can:
- Drop columns
- Handle missing values
- Encode categorical variables
- Normalize data
- Export cleaned CSV directly

👉 [Click here to open the Data Cleaning Tool on Google Colab](https://colab.research.google.com/drive/1SAMPLE_LINK)  
*(You only need to upload your file and choose cleaning options—no coding required!)*

""")


# '''dataset selection'''
st.sidebar.title("Upload CSV File")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file", type=["csv", "xlsx"], label_visibility="collapsed"
)

if not uploaded_file:
    st.info("Please upload a dataset to start Training of your model")
    st.stop()

try:
  if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df1 = pd.read_csv(uploaded_file)
    else:
        df1 = pd.read_excel(uploaded_file)

  st.sidebar.subheader("Select columns to visualize your data") 
  vis = st.sidebar.multiselect("Select columns to visualize",df1.columns.tolist(),label_visibility="collapsed",)
  df = df1[vis]
  

  st.sidebar.header("Train Test spliting parameters",divider="gray")
  st.sidebar.subheader("Select target variable(Y intercept)")
  Target = st.sidebar.selectbox("Target Variable", df1.columns.tolist(), label_visibility="collapsed")
  x = df1.drop(Target, axis=1)
  y = df1[Target]
   
  st.sidebar.subheader("choose the test size for your model")  
  test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
  st.sidebar.subheader("choose the random state for your model")  
  r_state = st.sidebar.number_input(
      "Please choose testing size",
      min_value=0,
      max_value=100,
      value=42,
      step=1,
      label_visibility="collapsed"
  )

  #train test split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=r_state)
  
  #main screen for displaying the data
  st.header("Data Preview")
  with st.expander("Brief overview", expanded=True):
     st.text("here you can see the rows of your data")
     st.text("unique data types in your data")
     st.text("Missing and duplicate values in your data")
     st.text("Statistical summary of your data")

  st.header("Quick Review",divider="gray")
  col1, col2, col3 = st.columns(3)

  with col1:
        st.metric("Total Missing Values", df1.isnull().sum().sum())

  with col2:
        st.metric("Total Duplicate Rows", df1.duplicated().sum())

  with col3:
        dtypes_list = df1.dtypes.astype(str).unique().tolist()
        st.write("**Data Types Present**")
        st.write(dtypes_list)

  st.header("Quick Analysis", divider="gray")
  st.dataframe(df1.head())

  left, right = st.columns(2)

  with left:
        st.subheader("Describe")
        st.dataframe(df1.describe())

  with right:
      st.subheader("Info")
      buffer = io.StringIO()
      df1.info(buf=buffer)
      s = "\n".join(buffer.getvalue().split("\n")[1:]) # Skip the first line
      st.text(s)


  # Visualize the data and encoding started here
  # Encode full df1 if no columns selected, else use selected df
  target_df = df if vis else df1
  encoded_df = target_df.copy()
  for col in encoded_df.select_dtypes(include='object').columns:
      if encoded_df[col].nunique() < 10:
          encoded_df[col] = encoded_df[col].astype('category').cat.codes
      else:
          le = LabelEncoder()
          encoded_df[col] = le.fit_transform(encoded_df[col].astype(str)) 

  # Visualize the data
  st.header("Quick Visualization",divider="gray")
  vis1, vis2 = st.columns(2)
  with vis1:
        st.subheader("Feature Distributions (Histogram)")        
        # Automatically fallback to all numerical columns if no columns are selected
        hist_cols = encoded_df[vis].select_dtypes(include='number').columns if vis else encoded_df.select_dtypes(include='number').columns    
        num_plots = len(hist_cols)
        cols = 3  # Number of columns per row in subplot grid
        rows = -(-num_plots // cols)  # Ceiling division        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()        
        for i, col in enumerate(hist_cols):
            axes[i].hist(encoded_df[col].dropna(), bins=20, edgecolor='black', color='skyblue')
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        fig.suptitle("Feature Distributions", fontsize=16)
        st.pyplot(fig)
    
  with vis2:
    st.subheader("Box Plot")

    # Automatically fallback to all numerical columns if no columns are selected
    box_cols = encoded_df[vis].select_dtypes(include='number').columns if vis else encoded_df.select_dtypes(include='number').columns
    fig, ax = plt.subplots(figsize=(10, 5))
    encoded_df[box_cols].plot(kind='box', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
  

  #model selection for data 
  st.header("Model Selection",divider="gray")
  st.subheader("Please select a Machine learning model to train your data")
  model = st.selectbox(
      "Select a model",
      (
          "None",
          "Logistic Regression",
          "Linear Regression",
          "Random Forest (Classifier)",
          "Decision Tree (Classifier)",
          "KNN (Classifier)",
          "SVM (Classifier)",
          "Random Forest (Regressor)",
          "Decision Tree (Regressor)",
          "KNN (Regressor)",
          "SVM (Regressor)",
      ),
      label_visibility="collapsed",
  )

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

  elif model == "Logistic Regression":
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
  
  elif model == "Linear Regression":
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

  elif model == "Decision Tree (Classifier)":
        def objective(trial):
          params = {
              'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
              'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
              'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
              'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
          }
          model = DecisionTreeClassifier(**params, random_state=r_state)
          score = cross_val_score(model, x_train, y_train, cv=3, scoring='accuracy').mean()
          return score

        st.info("Tuning Decision Tree Classifier using Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        
        best_params = study.best_params
        st.success("Best hyperparameters found using Optuna!")
        st.json(best_params)
        
        # UI for user-selected final hyperparameters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            criterion = st.selectbox("criterion", ['gini', 'entropy', 'log_loss'], index=['gini', 'entropy', 'log_loss'].index(best_params['criterion']))
        with col2:
            max_depth = st.number_input("max_depth", min_value=1, max_value=100, value=best_params['max_depth'], step=1)
        with col3:
            min_samples_split = st.number_input("min_samples_split", min_value=2, max_value=20, value=best_params['min_samples_split'], step=1)
        with col4:
            min_samples_leaf = st.number_input("min_samples_leaf", min_value=1, max_value=20, value=best_params['min_samples_leaf'], step=1)
        
        if st.button("Train Final Model"):
            model_dtc = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=r_state
            )
            model_dtc.fit(x_train, y_train)
            y_pred = model_dtc.predict(x_test)
        
            st.subheader("Model Evaluation")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
        
            # Save model with joblib
            joblib_buffer = io.BytesIO()
            joblib.dump(model_dtc, joblib_buffer)
            st.download_button(
                label="Download Model (.joblib)",
                data=joblib_buffer.getvalue(),
                file_name="decision_tree_classifier.joblib",
                mime="application/octet-stream"
            )
        
            # Save model with pickle
            pickle_buffer = io.BytesIO()
            pickle.dump(model_dtc, pickle_buffer)
            st.download_button(
                label="Download Model (.pkl)",
                data=pickle_buffer.getvalue(),
                file_name="decision_tree_classifier.pkl",
                mime="application/octet-stream"
            )


  elif model == "Decision Tree (Regressor)":
        def objective(trial):
          params = {
              'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'poisson']),
              'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
              'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
              'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
          }
          model = DecisionTreeRegressor(**params, random_state=r_state)
          return cross_val_score(model, x_train, y_train, cv=3, scoring='r2').mean()

        st.info("Tuning Decision Tree Regressor using Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        
        best_params = study.best_params
        st.success("Best hyperparameters found using Optuna!")
        st.json(best_params)
        
        # Final Training Parameters Input
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            criterion = st.selectbox("criterion", ['squared_error', 'absolute_error', 'poisson'], index=['squared_error', 'absolute_error', 'poisson'].index(best_params['criterion']))
        with col2:
            max_depth = st.number_input("max_depth", min_value=1, max_value=100, value=best_params['max_depth'], step=1)
        with col3:
            min_samples_split = st.number_input("min_samples_split", min_value=2, max_value=20, value=best_params['min_samples_split'], step=1)
        with col4:
            min_samples_leaf = st.number_input("min_samples_leaf", min_value=1, max_value=20, value=best_params['min_samples_leaf'], step=1)
        
        if st.button("Train Final Model"):
            model_dtr = DecisionTreeRegressor(
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=r_state
            )
            model_dtr.fit(x_train, y_train)
            y_pred = model_dtr.predict(x_test)
        
            st.subheader("Model Evaluation")
            st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
            st.write(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")
        
            # Save with joblib
            joblib_buffer = io.BytesIO()
            joblib.dump(model_dtr, joblib_buffer)
            st.download_button(
                label="Download Model (.joblib)",
                data=joblib_buffer.getvalue(),
                file_name="decision_tree_regressor.joblib",
                mime="application/octet-stream"
            )
        
            # Save with pickle
            pickle_buffer = io.BytesIO()
            pickle.dump(model_dtr, pickle_buffer)
            st.download_button(
                label="Download Model (.pkl)",
                data=pickle_buffer.getvalue(),
                file_name="decision_tree_regressor.pkl",
                mime="application/octet-stream"
            )


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


  elif model == "SVM (Classifier)":
    # Scale data for SVM (important for classifiers)
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

except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

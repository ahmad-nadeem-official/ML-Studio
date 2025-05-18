import io
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# page configuration
st.set_page_config(page_title="ML Studio", page_icon="🤖", layout="wide")

st.title("ML Studio")
st.write(
    "This is a machine learning studio that allows you to upload a CSV file, visualize the data, and train a machine learning model."
)

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
  
  st.sidebar.subheader("Select target variable(Y intercept)")
  Target = st.sidebar.selectbox("Target Variable", df1.columns.tolist(), label_visibility="collapsed")
  x = df1.drop(Target, axis=1)
  y = df1[Target]
   
  st.sidebar.subheader("choose the test size for your model")  
  test_size = st.sidebar.number_input(
      "Please choose testing size",
      min_value=0.1,
      max_value=1.0,
      value=0.2,
      step=0.05,
      label_visibility="collapsed"
  )
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




except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

st.title("Federated Learning with Logistic Regression")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview:", data.head())
    
    if "TenYearCHD" not in data.columns:
        st.error("Dataset must contain 'TenYearCHD' column as the target variable.")
    else:
        # Preprocessing
        data.dropna(inplace=True)
        X = data.drop("TenYearCHD", axis=1)
        y = data["TenYearCHD"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        num_clients = st.slider("Number of Clients", 2, 50, 20)
        mal_fac = st.slider("Malicious Client Fraction", 0.0, 0.5, 0.2)
        num_epochs = st.slider("Number of Epochs", 10, 100, 50)
        learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.2)
        
        mal_clients = [i for i in range(int(num_clients * mal_fac))]
        
        # Split data for clients
        clients_data = np.array_split(pd.concat([X_train, y_train], axis=1), num_clients)
        
        def train_model(data):
            X = data.drop(columns="TenYearCHD")
            y = data["TenYearCHD"]
            model = LogisticRegression()
            model.fit(X, y)
            return model.coef_, model.intercept_
        
        # Initialize global model
        global_model = LogisticRegression()
        global_model.fit(X_train[:1], y_train[:1])  # Dummy fit to set attributes
        global_coefs = np.zeros_like(global_model.coef_)
        global_intercept = np.zeros_like(global_model.intercept_)
        
        acc_history = []
        
        for epoch in range(num_epochs):
            coefs_list = []
            intercept_list = []
            data_sizes = []
            
            for i, data_i in enumerate(clients_data):
                coefs, intercept = train_model(data_i)
                if i in mal_clients:
                    coefs = np.random.rand(*coefs.shape)
                    intercept = np.random.rand(*intercept.shape)
                
                coefs_list.append(coefs)
                intercept_list.append(intercept)
                data_sizes.append(len(data_i))
            
            # Federated averaging
            total_data = sum(data_sizes)
            global_coefs = sum((c * s for c, s in zip(coefs_list, data_sizes))) / total_data
            global_intercept = sum((i * s for i, s in zip(intercept_list, data_sizes))) / total_data
            
            global_model.coef_ = global_coefs
            global_model.intercept_ = global_intercept
            
            acc = global_model.score(X_test, y_test)
            acc_history.append(acc)
            
            st.write(f"Epoch {epoch + 1}: Accuracy = {acc:.4f}")
        
        # Plot training accuracy
        st.write("### Training Accuracy Over Epochs")
        fig, ax = plt.subplots()
        ax.plot(range(1, num_epochs + 1), acc_history, marker='o', linestyle='-')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training Performance")
        st.pyplot(fig)

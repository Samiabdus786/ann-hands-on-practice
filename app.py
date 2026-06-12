import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

st.set_page_config(page_title="ANN – Bank Churn Predictor", page_icon="🧠", layout="wide")

st.title("🧠 ANN – Bank Customer Churn Prediction")
st.markdown("An **Artificial Neural Network** trained on the Churn_Modelling dataset to predict whether a customer will leave the bank.")

# ── Sidebar: hyper-parameters ──────────────────────────────────────────────
st.sidebar.header("⚙️ Model Hyper-parameters")
epochs     = st.sidebar.slider("Epochs",        5,  100, 20)
batch_size = st.sidebar.slider("Batch Size",    8,  128, 32, step=8)
h1_neurons = st.sidebar.slider("Hidden Layer 1 Neurons", 4, 128, 16, step=4)
h2_neurons = st.sidebar.slider("Hidden Layer 2 Neurons", 4, 64,  8,  step=4)

# ── Data upload ─────────────────────────────────────────────────────────────
st.subheader("📂 Upload Dataset")
uploaded = st.file_uploader("Upload Churn_Modelling.csv", type=["csv"])

@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file)
    X  = df[["CreditScore", "Age", "Balance", "EstimatedSalary"]].values
    y  = df["Exited"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, df

if uploaded:
    X_train, X_test, y_train, y_test, scaler, df = load_and_preprocess(uploaded)
    st.success(f"Dataset loaded – {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("🚀 Train ANN Model"):
        with st.spinner("Training …"):
            model = keras.Sequential([
                keras.layers.Input(shape=(4,)),
                keras.layers.Dense(h1_neurons, activation="relu"),
                keras.layers.Dense(h2_neurons, activation="relu"),
                keras.layers.Dense(1,           activation="sigmoid"),
            ])
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            history = model.fit(X_train, y_train, epochs=epochs,
                                batch_size=batch_size, validation_split=0.2, verbose=0)

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        st.success(f"✅ Test Accuracy: **{accuracy*100:.2f}%** | Test Loss: **{loss:.4f}**")

        # ── Training curves ───────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.plot(history.history["accuracy"],     label="Train Accuracy")
            ax.plot(history.history["val_accuracy"], label="Val Accuracy")
            ax.set_title("Accuracy over Epochs"); ax.legend()
            st.pyplot(fig)
        with col2:
            fig2, ax2 = plt.subplots()
            ax2.plot(history.history["loss"],     label="Train Loss")
            ax2.plot(history.history["val_loss"], label="Val Loss")
            ax2.set_title("Loss over Epochs"); ax2.legend()
            st.pyplot(fig2)

        st.session_state["ann_model"]  = model
        st.session_state["ann_scaler"] = scaler

# ── Single prediction ────────────────────────────────────────────────────────
st.subheader("🔮 Predict for a Single Customer")
with st.form("predict_form"):
    c1, c2, c3, c4 = st.columns(4)
    credit  = c1.number_input("Credit Score",      300, 900, 650)
    age     = c2.number_input("Age",               18,  92,  35)
    balance = c3.number_input("Balance",           0.0, 300000.0, 50000.0)
    salary  = c4.number_input("Estimated Salary",  0.0, 250000.0, 80000.0)
    submit  = st.form_submit_button("Predict")

if submit:
    if "ann_model" not in st.session_state:
        st.warning("⚠️ Please train the model first.")
    else:
        inp  = np.array([[credit, age, balance, salary]])
        inp  = st.session_state["ann_scaler"].transform(inp)
        prob = st.session_state["ann_model"].predict(inp, verbose=0)[0][0]
        label = "🚨 Will Churn" if prob > 0.5 else "✅ Will Stay"
        st.metric("Prediction", label, f"Confidence: {max(prob, 1-prob)*100:.1f}%")

st.markdown("---")
st.markdown("**Architecture:** Input(4) → Dense(ReLU) → Dense(ReLU) → Dense(Sigmoid)")

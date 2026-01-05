import streamlit as st
import numpy as np
import joblib

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Country Clustering App",
    page_icon="🌍",
    layout="centered"
)

# ---------------- Load model & scaler ----------------
@st.cache_resource
def load_model():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return kmeans, scaler

kmeans, scaler = load_model()

# ---------------- Title ----------------
st.title("🌍 Country Development Clustering")
st.write("Predict the cluster based on socio-economic indicators")

# ---------------- User Inputs ----------------
st.header("Enter Country Indicators")

gdp = st.number_input("GDP per Capita", min_value=500.0, max_value=100000.0, value=20000.0)
internet = st.number_input("Internet Usage (%)", min_value=0.0, max_value=100.0, value=60.0)

# ---------------- Prediction ----------------
if st.button("Predict Cluster"):
    input_data = np.array([[gdp, internet]])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)

    st.success(f"🌐 This country belongs to **Cluster {cluster[0]}**")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("K-Means Clustering | Streamlit Deployment")
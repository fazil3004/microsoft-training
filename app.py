import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------------
# App Title and Description
# -------------------------------
st.title("💻 Laptop Price Prediction")
st.write("This application predicts laptop prices using a trained Machine Learning model.")

# -------------------------------
# Load Saved Model and Data
# -------------------------------
try:
    ml_model = pickle.load(open("pipe.pkl", "rb"))
    df = pickle.load(open("df.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ Model or data file not found. Please ensure 'pipe.pkl' and 'df.pkl' are in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# -------------------------------
# User Inputs
# -------------------------------
co = st.selectbox("Select Company", df["Company"].unique(), index=3)
ty = st.selectbox("Select Type", df["TypeName"].unique(), index=3)
cpu = st.selectbox("Select CPU", df["Cpu"].unique(), index=3)
ram = st.radio("Select RAM (in GB)", [8, 16, 32, 64])
gpu = st.selectbox("Select GPU", df["Gpu"].unique(), index=3)
os = st.selectbox("Select Operating System", df["OpSys"].unique(), index=3)

we = st.slider("Weight of Laptop (kg)", min_value=1.0, max_value=4.5, value=2.0, step=0.1)
sp = st.slider("CPU Speed (GHz)", min_value=0.5, max_value=5.0, value=2.5, step=0.1)
cpu_type = st.slider("Clock Speed of Processor (GHz)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
ppi = st.slider("Pixel Density (PPI)", min_value=75, max_value=400, value=150, step=5)

ips = st.radio("Does the laptop have an IPS display?", ["yes", "no"], index=1)
touchscreen = st.radio("Does the laptop have a touchscreen?", ["yes", "no"], index=1)

hdd = st.selectbox("HDD Size (GB) — select 0 if only SSD", [0, 512, 1024, 2048], index=1)
ssd = st.selectbox("SSD Size (GB) — select 0 if only HDD", [0, 512, 1024, 2048], index=1)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("💰 Predict Laptop Price"):
    # Convert categorical to binary values
    ips = 1 if ips == "yes" else 0
    touchscreen = 1 if touchscreen == "yes" else 0

    # Arrange input in same order as model was trained
    query = np.array([[co, ty, cpu, ram, gpu, os, we, ips, touchscreen, sp, hdd, ssd, ppi]], dtype=object)

    try:
        op = ml_model.predict(query)
        # If model predicts log(price), convert back
        price = np.exp(op[0]) if op[0] < 15 else op[0]

        st.subheader("💸 Estimated Price:")
        st.success(f"₹ {int(price):,}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")



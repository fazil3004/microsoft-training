import streamlit as st
import pickle
import numpy as np

st.title("Laptop Price Prediction")
st.text("This App Is Created Using mL Model")
ml_model = pickle.load(open("pipe.pkl","rb"))
df = pickle.load(open("df.pkl","rb"))
df.head()

co = st.selectbox("Select Company", df["Company"].unique(), index=3)
ty = st.selectbox("Select Type", df["TypeName"].unique(), index=3)
cpu = st.selectbox("Select CPU", df["Cpu"].unique(), index=3)
ram = st.radio("RAM ", [8,16,32,64])
gpu = st.selectbox("Select GPU", df["Gpu"].unique(), index=3)
os = st.selectbox("Select OS", df["OpSys"].unique(), index=3)
we = st.slider("Weight Of Lap", min_value=1.0, max_value=4.5, value=2.0, step=0.1)
sp = st.slider("CPU Speed", min_value=0.0, max_value=4.0, value=2.0, step=0.1)
ips = st.radio("Does the laptop have ips display", ["yes","no"], index=1)
touchscreen = st.radio("Does the laptop have touchscreen", ["yes","no"], index=1)

# ❌ Wrong before: selectbox used with min_value/max_value — fixed to slider
cpu_type = st.slider("What is the clock speed of the processor (in GHz)", min_value=0.0, max_value=4.5, value=2.0, step=0.1)

# ❌ Wrong before: st("...") — fixed to st.selectbox
hdd = st.selectbox("HDD size (in GB). Select 0 if system only has SSD storage", [0,512,1024,2048])
ssd = st.selectbox("SSD size (in GB). Select 0 if system only has HDD storage", [0,512,1024,2048])

ppi = st.slider("What is the ppi (pixel density)", min_value=75, max_value=400, value=150, step=5)

if st.button("Predict Price"):
  if ips == "yes":
    ips = 1
  else:
    ips = 0

  if touchscreen == "yes":
    touchscreen = 1
  else:
    touchscreen = 0

  # make sure variable names match your inputs
  query = np.array([[co, ty, cpu, ram, gpu, os, we, ips, touchscreen, sp, hdd, ssd, ppi]])
  
  # ❌ Wrong before: used == instead of =
  op = ml_model.predict(query)

  # ❌ Wrong before: st.shoulder — fixed to st.subheader
  st.subheader("The estimated price of the laptop with the above mentioned specification is:")
  st.success(op[0])


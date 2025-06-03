import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Stres Tahmin Uygulaması", layout="centered")
st.title("🧠 Stres Tahmin Uygulaması")
st.markdown("Bu uygulama, belirli psikolojik ve davranışsal ölçütlere göre kişinin stresli olup olmadığını tahmin eder.")

features = ['cesd', 'mbi_ex', 'mbi_ea', 'health', 'mbi_cy']

st.sidebar.header("🔧 Girdi Değerleri")
user_input = {}
for feature in features:
    user_input[feature] = st.sidebar.slider(feature, min_value=0, max_value=100, value=50)

input_df = pd.DataFrame([user_input])

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

st.subheader("📊 Tahmin Sonucu:")
if prediction == 1:
    st.error("🔴 Tahmin: **Stresli**")
else:
    st.success("🟢 Tahmin: **Stresli Değil**")

st.markdown("---")
st.caption("Model: KNN (Korelasyon ile seçilen 5 özellik ile eğitildi)")

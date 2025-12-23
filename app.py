import streamlit as st
import joblib
import pandas as pd

# Konfigurasi dasar halaman
st.set_page_config(
    page_title="Prediksi Pembelian Produk",
    page_icon="🛍️",
    layout="wide",
)

# Muat model dan parameter scaling
gnb = joblib.load("naive_bayes_model.pkl")
scaling_params = joblib.load("scaling_params.pkl")

# Gaya kustom untuk tampilan lebih modern
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 50%, #e0f7fa 100%);
    }
    .block-container {
        padding: 2rem 2.5rem 3rem;
    }
    .hero {
        background: #0f172a;
        color: white;
        padding: 1.5rem 1.75rem;
        border-radius: 14px;
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.25);
    }
    .card {
        background: white;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
    }
    .result-pill {
        display: inline-block;
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom: 0.2rem;">Prediksi Pembelian Produk</h1>
        <p style="margin: 0; opacity: 0.8;">
            Masukkan profil singkat pengguna untuk memprediksi kemungkinan pembelian.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Input Data Pengguna")

# Form input untuk data pengguna
gender = st.sidebar.selectbox("Pilih Gender", ["Female", "Male"])
age = st.sidebar.slider("Usia Pengguna", 18, 100, 30)
salary = st.sidebar.slider("Gaji Pengguna (per tahun)", 0, 200000, 50000, step=1000)

# Encoding Gender
gender_encoded = 1 if gender == "Male" else 0

# Scaling sesuai dengan data training
age_scaled = (age - scaling_params["mean_age"]) / scaling_params["std_age"]
salary_scaled = (salary - scaling_params["mean_salary"]) / scaling_params["std_salary"]

# DataFrame input
input_data = pd.DataFrame(
    {
        "Gender": [gender_encoded],
        "Age": [age_scaled],
        "EstimatedSalary": [salary_scaled],
    }
)

# Prediksi kelas dan probabilitas
probabilities = gnb.predict_proba(input_data)
threshold = 0.5
prediction = 1 if probabilities[0][1] >= threshold else 0

prob_purchase = probabilities[0][1] * 100
prob_no_purchase = probabilities[0][0] * 100

# Layout dua kolom: hasil & rincian
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Hasil Prediksi")
    if prediction == 1:
        st.markdown(
            '<span class="result-pill" style="background:#dcfce7;color:#14532d;">'
            "Pembelian Produk (Ya)"
            "</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="result-pill" style="background:#fee2e2;color:#7f1d1d;">'
            "Tidak Membeli Produk"
            "</span>",
            unsafe_allow_html=True,
        )

    st.write(" ")
    st.write("Probabilitas:")
    st.progress(prob_purchase / 100)
    st.write(f"Peluang Pembelian: **{prob_purchase:.2f}%**")
    st.write(f"Peluang Tidak Pembelian: **{prob_no_purchase:.2f}%**")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Ringkasan Input")
    st.metric("Gender", gender)
    st.metric("Usia", f"{age} tahun")
    st.metric("Gaji (per tahun)", f"${salary:,.0f}")
    st.caption(
        "Nilai sudah disesuaikan dengan skala data pelatihan agar model lebih akurat."
    )
    st.markdown("</div>", unsafe_allow_html=True)

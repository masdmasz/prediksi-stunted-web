import pandas as pd
from unicodedata import name
import streamlit as st
import joblib
import numpy as np

# 1. Load model yang sudah kita simpan di Langkah 2
try:
    model = joblib.load('model_stunting_gk.pkl')
except:
    st.error("File 'model_stunting_rf.pkl' tidak ditemukan! Jalankan Langkah 2 terlebih dahulu.")

# 2. Tampilan Web App
st.markdown("## <center> 🏥 Website Prediksi Penderita Stunting 🏥 </center>", unsafe_allow_html=True)
st.write("Masukkan data pasien untuk melakukan pengecekan risiko stunting.")

# 3. Form Input Data Medis Pasien
st.write("### Data Klinis")
nama = st.text_input("Nama Pasien", value="")
age = st.number_input("Umur Balita (Bulan)", min_value=0, max_value=60, value=0)
gender = st.selectbox("Jenis Kelamin", options=[("laki-laki"), ("perempuan")])
if gender == "laki-laki":
    gender = 0
else:
    gender = 1
tinggi_badan = st.number_input("Tinggi Badan (cm)", value=0.0)

# 4. Logika Tombol Prediksi
if st.button("Mulai Diagnosa"):
    # Susun data input menjadi format yang dipahami AI
# Tambahkan dtype=object di ujung array agar Python mengizinkan campuran teks dan angka
    data_pasien = pd.DataFrame([{
        "Umur (bulan)": age,
        "Jenis Kelamin": gender,
        "Tinggi Badan (cm)": tinggi_badan
    }])

# Baru lakukan prediksi
    hasil_prediksi = model.predict(data_pasien)
    st.subheader("Hasil Analisis:")
    if hasil_prediksi[0].lower() == 'stunted' or hasil_prediksi[0].lower() == 'severely stunted':
        st.error("⚠️ Hasil: Balita Terprediksi STUNTING. Disarankan pemeriksaan klinis lebih lanjut.") 
    elif hasil_prediksi[0].lower() == 'tinggi':
        st.warning("Hasil: Balita Terprediksi TIDAK STUNTING (Tinggi).")
    else:
        st.success("✅ Hasil: Balita Terprediksi TIDAK STUNTING (Normal).")
    
st.write("---")
st.caption("📢 **Catatan/Disclaimer:** Aplikasi ini menggunakan kecerdasan buatan dengan akurasi model sebesar 99,52% untuk keperluan skrining awal akademis. Hasil prediksi sistem bukan merupakan diagnosis final medis resmi. Harap selalu konsultasikan hasil laboratorium Anda kepada dokter atau tenaga medis ahli.")
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

os = os.system('cls' if os.name == 'nt' else 'clear')

print("START TRAINING...")

# LOAD DATASET

df = pd.read_csv("data_balita.csv")

print("DATASET BERHASIL DIBACA")
print(df.head())

# ENCODE GENDER
le_gender = LabelEncoder()

df["Jenis Kelamin"] = le_gender.fit_transform(
    df["Jenis Kelamin"]
)

# FEATURE & TARGET
X = df[
    [
        "Umur (bulan)",
        "Jenis Kelamin",
        "Tinggi Badan (cm)",
    ]
]

y = df["Status Gizi"]

print("FEATURE SIAP")

# ============================================
# SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("SPLIT 1000 DATA BERHASIL")

# ============================================
# MODEL
# ============================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

print("TRAINING MODEL...")

model.fit(X_train, y_train)

print("MODEL SELESAI TRAINING")

# ============================================
# EVALUATION
# ============================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ============================================
# SAVE MODEL
# ============================================

joblib.dump(
    model,
    "model_stunting_gk.pkl"
)

print("\nMODEL BERHASIL DISIMPAN")

classes = model.classes_

# 7. Layout dan Tampilkan Hasil Keberhasilan Model
title = f'AKURASI MODEL : {accuracy_score(y_test, y_pred) * 100:.2f}%'
width = 55

line = '✦ ' + '=' * (width - 6) + ' ✦'

print('\n')
print(line)
print(title.center(width - 4))
print(line)
print("LAPORAN KLASIFIKASI:")
print(classification_report(y_test, y_pred, target_names=classes))
print("\nMATRIKS KEBERHASILAN:")
print(confusion_matrix(y_test, y_pred, labels=classes))
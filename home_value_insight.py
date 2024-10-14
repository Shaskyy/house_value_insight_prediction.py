import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Judul Aplikasi
st.title('Analisis Regresi Harga Rumah')

# Mengunggah File CSV
uploaded_file = st.file_uploader("Upload dataset CSV", type="csv")
if uploaded_file is not None:
    # Membaca file csv & disimpan ke DataFrame bernama 'dfHome'
    dfHome = pd.read_csv(uploaded_file)
    st.write(dfHome.head())  # Menampilkan 5 baris pertama

    # Memisahkan fitur dan target
    x = dfHome.drop(columns=['House_Price']) 
    y = dfHome['House_Price']  

    # Membagi Data Menjadi Data Training & Data Testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Melatih Model
    model_home = LinearRegression()
    model_home.fit(x_train, y_train)

    # Memprediksi Nilai pada Data Testing
    y_prediksi = model_home.predict(x_test)

    # Menampilkan Metrik Evaluasi
    st.write('MAE : ', mean_absolute_error(y_test, y_prediksi))
    st.write('R2  : ', r2_score(y_test, y_prediksi))

    # Data Baru untuk Prediksi
    new_data = np.array([
        [1000, 2, 1, 2001, 4.7, 1, 9],  # Data 1
        [2500, 2, 1, 2002, 5.2, 2, 3],  # Data 2
        [1904, 5, 3, 2003, 3.1, 1, 4],  # Data 3
        [3300, 3, 2, 2004, 4.2, 2, 7],  # Data 4
        [420, 1, 1, 2005, 1.0, 0, 1],   # Data 5
        [5900, 5, 3, 2006, 2.2, 2, 4],  # Data 6
        [3100, 3, 1, 2007, 8.2, 1, 5],  # Data 7
        [1201, 2, 1, 2008, 7.1, 0, 1],  # Data 8
        [700, 1, 1, 2009, 0.5, 0, 1],   # Data 9
        [3100, 4, 2, 2010, 6.3, 2, 6]   # Data 10
    ])
    new_prediksi = model_home.predict(new_data)

    # Visualisasi Hasil Prediksi untuk Data Testing
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_prediksi, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
    plt.xlabel('Harga Asli')
    plt.ylabel('Harga Prediksi')
    plt.title('Hasil Prediksi Data Testing')
    plt.grid()
    st.pyplot(plt)  # Menampilkan plot dengan Streamlit

    # Visualisasi Hasil Prediksi untuk Data Testing dan Data Baru
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_prediksi, alpha=0.6, label='Data Testing')
    sns.scatterplot(x=new_prediksi, y=new_prediksi, color='yellow', s=100, label='Data Baru')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
    plt.xlabel('Harga Asli')
    plt.ylabel('Harga Prediksi')
    plt.title('Visualisasi Hasil Prediksi - Data Testing dan Data Baru')
    plt.legend()
    plt.grid()
    st.pyplot(plt)  # Menampilkan plot dengan Streamlit

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.utils.validation import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import pickle
from pickle import dump 
from collections import OrderedDict

st.sidebar.title("Selamat Datang!")
st.sidebar.write("Di Website Prediksi Gagal Ginjal Menggunakan Metode Naive Bayes Gaussian.")


page1, page2, page3, page4, page5 = st.tabs(["Home", "Data", "Preprocessing", "Input Model", "Prediksi"])

with page1:
    st.title("Klasifikasi Prediksi Gagal Ginjal Menggunakan Metode Naive Bayes Gaussian")
    st.write("Dataset Yang digunakan adalah **Gagal Ginjal** dari [Kaggle](https://www.kaggle.com/datasets/abhia1999/chronic-kidney-disease)")
    st.write("Link repository Github : [https://github.com/dimasmhrdka/web-pendat](https://github.com/dimasmhrdka/web-pendat)")
    st.header("Deskripsi Data")
    st.write("""
        Dataset yang digunakan adalah dataset tentang Gagal Ginjal untuk memprediksi secara diagnostik apakah seorang pasien menderita penyakit ginjal kronis atau tidak, 
        berdasarkan pengukuran diagnostik tertentu yang disertakan dalam kumpulan data. Yang memiliki kolom kelas dengan nama Class nilai nya antara 0 atau 1. 
        Jika 1 berarti pasien tersebut menderita penyakit ginjal kronis, kemudian jika 0 berarti pasien tersebut tidak menderita penyakit ginjal kronis. 
        Untuk fiturnya sendiri ada 13 yaitu : 'Bp', 'Sg', 'Al', 'Su', 'Rbc', 'Bu', 'Sc', 'Sod', 'Pot', 'Hemo', 'Wbcc', 'Rbcc', dan 'Htn'.
        """)

with page2:
    st.title("Dataset Gagal Ginjal")
    data = pd.read_csv("https://raw.githubusercontent.com/dimasmhrdka/data-csv/main/new_model.csv")
    deleteCol = data.drop(["Class"], axis=1)
    st.write(deleteCol)

with page3:
    st.title("Halaman PreProcessing")
    st.write("Preprocessing data merupakan tahapan untuk melakukan mining data sebelum tahap pemrosesan. fungsi preprocessing data untuk mengubah data mentah menjadi data yang mudah dipahami.")
    st.markdown("""
        <ol>
            <li>Data Cleaning</li>
            <li>Transformasi Data</li>
            <li>Mengurangi Data</li>
        </ol>
    """, unsafe_allow_html=True)
    st.write("Disini preprocessing menggunakan transformasi data dengan metode MinMaxScaller()")
    st.write("MinMaxScaler() merupakan transformasi data dengan rentang tertentu, rentang yang digunakan disini yaitu 0 - 1. Rumus transformasi data dapat menggunakan berikut:")
    rumus1 = '''X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))'''
    rumus2 = '''X_scaled = X_std * (max - min) + min'''
    st.code(rumus1, language="python")
    st.code(rumus2, language="python")
    st.write("Sebelum itu harus di encoder dulu data attribut yang memiliki type categorial harus diubah menjadi type numerik agar bisa dilakukan preprocessing.")

    st.subheader("Split Data")
    codeSplit = '''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)'''
    st.code(codeSplit, language="python")
    st.write("Split Data disini digunakan untuk memisahkan data menjadi nilai acak. data yang digunakan sebagai data testing sebesar 0.3 atau 30% dari data yang digunakan. sehingga sisanya digunakan sebagai data training.")



    creditScoreRaw = pd.read_csv("https://raw.githubusercontent.com/dimasmhrdka/data-csv/main/new_model.csv")

    labels = data["Class"]

    # create a dataframe with all training data except the target column
    X = data.drop(columns=["Class"])

    # check that the target variable has been removed
    X.head()

    # Preprocessing
    old_normalize_feature_labels = ['Bp', 'Sg', 'Al', 'Su', 'Rbc', 'Bu', 'Sc', 'Sod', 'Pot', 'Hemo', 'Wbcc', 'Rbcc', 'Htn']
    new_normalized_feature_labels = ['Bp', 'Sg', 'Al', 'Su', 'Rbc', 'Bu', 'Sc', 'Sod', 'Pot', 'Hemo', 'Wbcc', 'Rbcc', 'Htn']
    normalize_feature = data[old_normalize_feature_labels]

    scaler = MinMaxScaler()
    scaler.fit(normalize_feature)
    MinMaxScaler()

    normalized_feature = scaler.transform(normalize_feature)
    normalized_feature_data = pd.DataFrame(normalized_feature, columns = new_normalized_feature_labels)

    X = X.drop(columns = old_normalize_feature_labels)
    X = X.join(normalized_feature_data)
    X = X.join(labels)

    dump(scaler, open('scaler.save', 'wb'))

    percent_amount_of_test_data = 0.3

    # separate target 

    # values
    matrices_X = X.iloc[:,0:13].values

    # classes
    matrices_Y = X.iloc[:,13].values

    X_1 = X.iloc[:,0:13].values
    Y_1 = X.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X_1, Y_1, test_size = percent_amount_of_test_data, random_state=0)

    ### Dictionary to store model and its accuracy

    model_accuracy = OrderedDict()

    ### Dictionary to store model and its precision

    model_precision = OrderedDict()

    ### Dictionary to store model and its recall

    model_recall = OrderedDict()

    ### Applying Naive Bayes Classification model

    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(X_train, y_train)
    Y_pred_nb = naive_bayes_classifier.predict(X_test)

    naive_bayes_accuracy = round(100 * accuracy_score(y_test, Y_pred_nb), 2)
    model_accuracy['Gaussian Naive Bayes'] = naive_bayes_accuracy

    naive_bayes_precision = round(100 * precision_score(y_test, Y_pred_nb, average = 'weighted'), 2)
    model_precision['Gaussian Naive Bayes'] = naive_bayes_precision

    naive_bayes_recall = round(100 * recall_score(y_test, Y_pred_nb, average = 'weighted'), 2)
    model_recall['Gaussian Naive Bayes'] = naive_bayes_recall

    filename = 'model.joblib'
    joblib.dump(naive_bayes_classifier, filename)

    # load the model from disk
    loaded_model = joblib.load(filename)
    result = loaded_model.score(X_test, y_test)

    st.write(result)

with page4:
    st.title("Input Data Model")

    # membuat input
    bp = st.text_input("Blood Pressure (1-280)")
    sg = st.text_input("Specific Gravity (1.01-1.03)")
    al = st.text_input("Albumin (0-10)")
    su = st.text_input("Sugar(0-10)")
    rbc = st.text_input("Red Blood Cell (0 & 1)")
    bu = st.text_input("Blood Urea  (1-391)")
    sc = st.text_input("Serun Creatnine (0-76)")
    sod = st.text_input("Sodium (1-163)")
    pot = st.text_input("Pottasium (1-50)")
    hemo = st.text_input("Hemoglobin (3-19)")
    wbcc = st.text_input("White Blood Cc (1000-30000)")
    rbcc = st.text_input("Red Blood Cc (1-10)")
    htn = st.text_input("Hypertension (0 & 1)")

    # section output
    def submit():

        scaler = joblib.load("scaler.save")
        normalize = scaler.transform([[int(bp),int(sg),int(al),int(su),int(rbc),int(bu),int(sc),int(sod),int(pot),int(hemo),int(wbcc),int(rbcc),int(htn)]])[0].tolist()

        # create data input
        data_input = {
            "bp" : normalize[0],
            "sg" : normalize[1],
            "al" : normalize[2],
            "su" : normalize[3],
            "rbc" : normalize[4],
            "bu" : normalize[5],
            "sc" : normalize[6],
            "sod" : normalize[7],
            "pot" : normalize[8],
            "hemo" : normalize[9],
            "wbcc" : normalize[10],
            "rbcc" : normalize[11],
            "htn" : normalize[12],
        }

        inputs = np.array([[val for val in data_input.values()]])

        # filenameModel = "model.joblib"
        # joblib.dump(naive_bayes_classifier, filename)

        model = joblib.load("model.joblib")

        # predAkurasi = model.score()
        # with open("model.sav", "rb") as model_buffer:
            # model = pickle.load(model_buffer)
        pred = model.predict(inputs)
        # returnData = [pred, predAkurasi]
        return pred

    # create button submit
    submitted = st.button("Prediksi")
    if submitted:
        st.text("Hasil prediksi ada pada halaman Prediksi")
        with page5:
            st.write("Hasil prediksi risk rating yang di peroleh yaitu:")
            st.text(submit())
            # st.write("akurasi data uji")
            # predAkurasi = submit()[1].score()
            # st.text(predAkurasi)

with page5:
    if not submitted:
        st.write("Belum ada prediksi, Harap masukkan input model dahulu!")
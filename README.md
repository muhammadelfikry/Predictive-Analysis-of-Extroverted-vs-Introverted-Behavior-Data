# Laporan Proyek Machine Learning - Muhammad Elfikry

## Domain Proyek

**Latar Belakang:**

*Personality classification* merupakan salah satu aplikasi penting dalam bidang psikologi, pemasaran, pendidikan, dan sumber daya manusia. Salah satu pendekatan yang sering digunakan dalam klasifikasi kepribadian adalah dengan membedakan antara *introvert* dan *extrovert*. Kepribadian ini mempengaruhi cara seseorang berperilaku, membuat keputusan, dan berinteraksi dengan lingkungan sosialnya.

Dengan meningkatnya aktivitas daring dan penggunaan media sosial, terdapat peluang besar untuk mengamati perilaku digital seseorang dan menggunakannya sebagai indikator tipe kepribadian. Oleh karena itu, membangun model *machine learning* untuk memprediksi kepribadian berdasarkan perilaku dapat membantu banyak organisasi dalam memahami pengguna atau klien mereka.

**Mengapa masalah ini penting?**
- Dalam HR dan rekrutmen, dapat membantu dalam *job matching*.
- Dalam pendidikan, membantu menyesuaikan gaya belajar.
- Dalam pemasaran, meningkatkan personalisasi konten.

**Referensi:**
1. Cain, S. (2012). Quiet: The Power of Introverts in a World That Can't Stop Talking.
2. Grant, A. M. (2013). Give and Take: A Revolutionary Approach to Success.
3. Amichai-Hamburger, Y. (2002). Internet and personality. Computers in Human Behavior, 18(1), 1–10. https://doi.org/10.1016/S0747-5632(01)00034-6

## Business Understanding

### Problem Statements

- Bagaimana mengklasifikasikan apakah seseorang termasuk tipe *introvert* atau *extrovert* berdasarkan perilaku digital atau aktivitas kesehariannya?
- Fitur apa yang paling berpengaruh dalam membedakan antara *introvert* dan *extrovert*?
- Seberapa baik performa model klasifikasi yang dibangun dalam memprediksi tipe kepribadian tersebut?

### Goals

- Membangun model klasifikasi *machine learning* untuk membedakan *introvert* dan *extrovert*.
- Menemukan fitur penting dari data perilaku yang berkontribusi signifikan dalam klasifikasi kepribadian.
- Mengevaluasi performa model menggunakan metrik klasifikasi seperti *akurasi, precision, recall*, dan *F1-score*.

### Solution statements
- Mengembangkan model prediktif menggunakan beberapa algoritma, termasuk *Support Vector Machine, K-Nearest Neighbors*, dan *Decision Tree*.
- Melakukan *Exploratory Data Analysis* (EDA) untuk memahami distribusi data dan pengaruh masing-masing fitur terhadap target.
- Melakukan *hyperparameter tuning* pada model terbaik guna meningkatkan akurasi prediksi.
- Mengevaluasi performa model menggunakan *confusion matrix*.

## Data Understanding
*Dataset* yang digunakan berasal dari Kaggle dan dapat diakses melalui tautan berikut:
[Kaggle](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data).

Dataset ini terdiri dari 2,900 baris data dengan tujuh fitur dan satu label target yang menunjukkan apakah seseorang adalah *extrovert* atau *introvert*. Dataset yang digunakan mengandung 458 *missing values* dan 388 entri duplikat, yang perlu ditangani untuk memastikan kualitas data yang optimal sebelum pelatihan model. Pemeriksaan terhadap *missing value* dan data duplikat dilakukan menggunakan kode berikut:

```python
print("Total missing value: ", data.isnull().sum().sum())
print("Total duplicate: ", data.duplicated().sum())
```

output:

```text
Total missing value:  458
Total duplicate:  388
```

### Variabel-variabel pada Extrovert vs Introvert Behavior Data dataset:
- *Time_spent_Alone*: Jam yang dihabiskan sendirian setiap hari (0–11).
- *Stage_fear*: Keberadaan demam panggung (*Yes/No*).
- *Social_event_attendance*: Frekuensi menghadiri acara sosial (0–10).
- *Going_outside*: Frekuensi pergi ke luar rumah (0–7).
- *Drained_after_socializing*: Merasa lelah setelah bersosialisasi (*Yes/No*).
- *Friends_circle_size*: Jumlah teman dekat (0–15).
- *Post_frequency*: Frekuensi posting di sosial media (0–10).
- *Personality*: Variabel target (*Extrovert/Introvert*).

### Exploratory Data Analysis (EDA)
- Visualisasi distribusi target class (*introvert vs extrovert*).
  
  ![image](https://github.com/user-attachments/assets/1207656b-98ef-47d6-a71b-45a3afe2ccf7)

  Gambar 1. Distribusi Target.
  
  Pada Gambar 1, Distribusi target pada dataset didominasi oleh label *Extrovert*.
   
- Visualisasi preferensi kegiatan (seperti *Social_event_attendance, Post_frequency*) antara *introvert* dan *extrovert*.
  
  ![image](https://github.com/user-attachments/assets/c16438b0-213f-434f-a958-d9e887d03e7f)

  Gambar 2. Hubungan fitur numerik terhadap target.

  Pada Gambar 2, kepribadian ekstrovert cenderung lebih sering berinteraksi sosial—seperti menghadiri acara, keluar rumah, memiliki lingkaran pertemanan besar, dan sering memposting—sementara orang introvert lebih banyak menghabiskan waktu sendirian.

- Visualisasi scatter plot untuk setiap pasangan fitur.

  ![image](https://github.com/user-attachments/assets/f75c1b9c-4732-4bff-bda8-c60673e4435a)

  Gambar 3. Scatter plot untuk setiap pasangan fitur.

  Pada gambar 3, terdapat perbedaan pola perilaku antara "Extrovert" dan "Introvert" dalam berbagai aspek interaksi sosial. Secara umum, ekstrover (biru) cenderung menunjukkan tingkat partisipasi yang lebih tinggi dalam aktivitas sosial seperti "Social_event_attendance", "Going_outside", dan "Friends_circle_size", serta mungkin memiliki "Post_frequency" yang lebih tinggi dan "Time_spent_Alone" yang lebih rendah dibandingkan dengan introver (hijau).
  
- Analisis korelasi antar fitur dengan *heatmap*.
  
  ![image](https://github.com/user-attachments/assets/3a826741-ed12-4397-8b87-a0862ca52194)

  Gambar 4. Korelasi fitur.

  Pada Gambar 4, Gambar heatmap menunjukkan bahwa semakin sering seseorang menghabiskan waktu sendirian, semakin rendah tingkat keaktifannya dalam aktivitas sosial, bergaul, dan frekuensi unggahan, dengan hubungan negatif yang kuat antar variabel tersebut.


## Data Preparation
**Beberapa tahapan yang dilakukan dalam proses data preparation:**
1. **Handling Missing and Duplicate Values**: Dataset dicek dan dibersihkan dari nilai kosong dan duplikat. Nilai kosong (*missing values*) dapat menyebabkan *error* saat pelatihan model atau membuat model belajar pola yang salah. Duplikat data dapat menyebabkan bias dalam pelatihan, karena model bisa terlalu mempelajari informasi yang berulang, sehingga mengurangi generalisasi terhadap data baru.
   
   ```python
   data.dropna(inplace=True)
   data.drop_duplicates(inplace=True)
   ```

   output:
   
   ```text
   Total missing value:  0
   Total duplicate:  0
   ```
   
3. **Label Encoding**: Fitur kategorikal seperti *Stage_fear* dan *Drained_after_socializing* diubah ke dalam bentuk numerik menggunakan teknik *One-hot Encoding*. *One-hot Encoding* digunakan untuk menghindari asumsi urutan atau besaran pada fitur kategorikal. Teknik ini menciptakan kolom biner untuk setiap kategori, sehingga tidak ada informasi ordinal yang salah diterapkan oleh model.
   
   ```python
   feature_to_encode = ["Stage_fear", "Drained_after_socializing"]
   data_encoding = pd.get_dummies(data, columns=feature_to_encode)
   ```
   
5. **Split Dataset**: Dataset dibagi ke dalam tiga bagian: *train set, validation set*, dan *test set*, yang masing-masing digunakan untuk pelatihan, validasi, dan pengujian model.
   
   ```python
   x = data_encoding.drop(["Personality"], axis=1)
   y = data_encoding["Personality"]
    
   X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
   X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
   ```
   
7. **Feature Scaling**: Fitur numerik dinormalisasi menggunakan **MinMaxScaler** agar setiap nilai berada dalam rentang 0 hingga 1, sehingga meminimalkan skala perbedaan antar fitur. *MinMaxScaler* mengubah nilai menjadi rentang 0–1, yang membantu mempercepat konvergensi saat pelatihan dan mencegah fitur dengan nilai besar mendominasi proses pembelajaran.
   
   ```python
   min_max_scaler = MinMaxScaler()
   
   X_train = min_max_scaler.fit_transform(X_train)
   X_test = min_max_scaler.transform(X_test)
   X_val = min_max_scaler.transform(X_val)
   ```

## Modeling
Model *Machine Learning* yang digunakan untuk menyelesaikan permasalahan ini adalah sebagai berikut:
1. Support Vector Machine (SVM)
   - Digunakan untuk menemukan hyperplane terbaik yang memisahkan kelas-kelas secara optimal.
   - Parameter: random_state=42
   - Kekurangan:
     - Kurang efisien untuk dataset besar (karena waktu komputasi tinggi).
     - Sulit untuk diinterpretasikan secara langsung.
     - Sensitif terhadap pemilihan parameter (seperti kernel, C, dan gamma).
     
3. K-Nearest Neighbors (KNN)
   - Memprediksi kelas berdasarkan mayoritas tetangga terdekat dalam ruang fitur.
   - Parameter: n_neighbors=10
   - Kekurangan:
     - Performanya bisa buruk pada dataset besar karena waktu prediksi lambat.
     - Sangat sensitif terhadap fitur yang tidak diskalakan (sehingga perlu *scaling*).
     - Pemilihan nilai k sangat mempengaruhi performa.
    
5. Decision Tree
   - Membangun struktur pohon keputusan berdasarkan fitur yang paling informatif secara bertahap.
   - Parameter: random_state=42
   - Kekurangan:
     - Rentan terhadap *overfitting*, terutama pada data kecil.
     - Keputusan sangat tergantung pada data latih (tidak stabil).
     - Performa bisa rendah jika tidak dilakukan pruning atau pengaturan parameter seperti *max_depth*.

Nilai akurasi yang diperoleh dari pelatihan model terhadap data training dan validation menunjukkan sejauh mana model mampu belajar dan menggeneralisasi pola dari data.

Tabel 1. Hasil akurasi dari pelatihan model terhadap data *training* dan *validation*.
    
| Model         | Train Accuracy | Validation Accuracy |
|---------------|----------------|---------------------|
| KNN           | 0.9118         | 0.919048            |
| Decision Tree | 0.97795        | 0.77619             |
| SVM           | 0.911204       | 0.919048            |

**Hyperparameter Tuning**

Dilakukan *tuning hyperparameter* menggunakan *GridSearchCV* pada model SVM karena model ini memiliki nilai *validation accuracy* terbaik dibandingkan dua model lainnya.

```python
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["auto", "scale", 0.01, 0.001],
    "kernel": ["linear", "rbf", "poly"]
}

grid_search = GridSearchCV(
    models["SVM"],
    param_grid,
    cv=5,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
```

Proses *GridSearchCV* dilakukan dengan *5-fold cross-validation* untuk mengevaluasi total 240 kombinasi *hyperparameter*.
Hasil terbaik diperoleh dengan parameter:

```text
{'C': 0.1, 'gamma': 'auto', 'kernel': 'linear'}
```
Model dengan parameter tersebut mencapai skor akurasi validasi sebesar 0.9190, yang hanya menunjukkan perubahan kecil dibandingkan sebelum dilakukan *hyperparameter tuning*.

## Evaluation
Evaluasi yang digunakan pada model SVM menggunakan *Classification Report* dan *Confusion Matrix* pada *Test Set*.
*Classification report* melakukan evaluasi berdasarkan metrik berikut:
- *Accuracy*: Tingkat keseluruhan prediksi yang benar.
- *Precision*: Ketepatan model dalam memprediksi kelas positif.
- *Recall*: Kemampuan model dalam menangkap semua data positif.
- *F1-Score*: Rata-rata harmonis antara *precision* dan *recall*.

Evaluasi model pada data uji menunjukkan performa akhir model dalam mengklasifikasikan data yang belum pernah dilihat sebelumnya. Berdasarkan hasil evaluasi pada data uji, model mencapai akurasi sebesar 94%, dengan nilai *precision, recall*, dan *f1-score* yang seimbang, menunjukkan performa generalisasi yang baik.

Tabel 2. Hasil evaluasi pada data uji.

| Class         | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| Extrovert     | 0.96      | 0.94   | 0.95     | 124     |
| Introvert     | 0.92      | 0.94   | 0.93     | 86      |
| **Accuracy**  |           |        | **0.94** | 210     |
| Macro Avg     | 0.94      | 0.94   | 0.94     | 210     |
| Weighted Avg  | 0.94      | 0.94   | 0.94     | 210     |

*Confusion matrix* yang divisualisasikan dalam bentuk heatmap memperlihatkan distribusi antara prediksi yang benar dan yang salah secara jelas, sehingga memudahkan dalam menginterpretasikan jenis kesalahan klasifikasi yang terjadi dengan hasil sebagai berikut:

![image](https://github.com/user-attachments/assets/6c6df1ed-2f18-4f18-95fb-af7d356ef1ed)

Gambar 4. *Confusion Matrix* hasil prediksi model pada data uji.

Pada Gambar 4, Model memiliki kinerja klasifikasi yang sangat baik dengan akurasi tinggi, terbukti dari jumlah prediksi benar yang jauh lebih besar dibandingkan kesalahan, yaitu hanya 7 *false positive* dan 5 *false negative* dari total 210 data.

**Kesimpulan Evaluasi**:

Model SVM menunjukkan performa yang sangat baik dalam membedakan antara kelas *Extrovert* dan *Introvert*, dengan nilai akurasi sebesar 94%. Nilai *precision* dan *recall* yang seimbang pada kedua kelas mengindikasikan bahwa model tidak hanya akurat, tetapi juga andal dalam menangani distribusi kelas yang mungkin tidak seimbang. Dengan *f1-score* yang tinggi, model ini dinilai cocok untuk digunakan pada data serupa dalam konteks klasifikasi kepribadian. Berdasarkan hasil *Exploratory Data Analysis* (EDA), fitur-fitur seperti:

- *Social_event_attendance*
- *Friends_circle_size*
- *Post_frequency*
- *Time_spent_Alone*

menunjukkan perbedaan distribusi yang signifikan antara kedua tipe kepribadian. Individu dengan kepribadian ekstrovert cenderung memiliki nilai tinggi pada fitur yang berkaitan dengan interaksi sosial, sedangkan individu dengan kepribadian introvert lebih tinggi pada waktu yang dihabiskan sendirian. Secara keseluruhan, hasil yang diperoleh menunjukkan bahwa pendekatan machine learning dapat dimanfaatkan secara efektif untuk mengidentifikasi tipe kepribadian berdasarkan perilaku. Temuan ini diharapkan dapat menjadi dasar bagi pengembangan model serupa di masa depan, dengan cakupan data dan konteks yang lebih luas.

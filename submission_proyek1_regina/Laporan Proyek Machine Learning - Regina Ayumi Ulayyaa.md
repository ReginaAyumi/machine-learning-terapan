# Laporan Proyek Machine Learning - Regina Ayumi Ulayyaa

## Domain Proyek

Pemasaran langsung merupakan salah satu strategi yang umum digunakan oleh berbagai industri, termasuk perbankan, untuk mencapai calon pelanggan secara langsung melalui komunikasi langsung seperti panggilan telepon, email, atau surat pos. Dalam konteks perbankan, tujuan utama dari kampanye pemasaran langsung adalah untuk meningkatkan penjualan produk atau layanan, seperti deposito berjangka, kartu kredit, atau pinjaman.

Data historis dari kampanye pemasaran langsung menyediakan sumber informasi yang berharga bagi perusahaan untuk menganalisis dan memahami perilaku calon pelanggan. Dengan mempelajari pola-pola dalam data ini, perusahaan dapat mengidentifikasi peluang-peluang untuk meningkatkan efektivitas kampanye mereka. 

Riset terkait telah dilakukan oleh Basarlan & Agun menggunakan algoritma klasifikasi K-Nearest Neighbor (KNN), Naive Bayes, and C4.5 Decision Tree serta menggunakan beberapa program berbeda untuk mengolah data tersebut. Algoritma yang memberikan hasil terbaik pada semua program adalah algoritma decision tree. Hasil ini menunjukkan bahwa metode Decision Tree memberikan kinerja yang lebih baik terlepas dari program yang digunakan [1].

Proyek ini akan menggunakan data terkait kampanye pemasaran langsung dari sebuah lembaga perbankan di Portugal sebagai studi kasus. Data ini mencakup informasi tentang kontak-kontak yang dilakukan kepada calon pelanggan melalui panggilan telepon dan hasil dari kontak tersebut. Salah satu pendekatan yang banyak digunakan adalah menggunakan teknik machine learning untuk membangun model prediktif yang dapat memprediksi keputusan pelanggan, seperti apakah mereka akan berlangganan produk atau tidak.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, berikut ini batasan masalah yang dapat diselesaikan dengan proyek ini:
- Bagaimana cara melakukan pemrosesan data yang optimal untuk meningkatkan performa model prediktif dalam memprediksi keputusan pelanggan untuk berlangganan deposito berjangka?
- Bagaimana mengevaluasi berbagai model machine learning yang berbeda untuk menentukan model yang paling sesuai dengan data dan mencapai akurasi tertinggi?

### Goals

- Memastikan data yang digunakan untuk melatih model prediktif diproses dengan baik dan siap untuk digunakan.
- Mendapatkan model machine learning yang paling cocok untuk memprediksi keputusan pelanggan dalam berlangganan deposito berjangka.

### Solution statements
- Preprocessing Data:
    - Melakukan identifikasi dan penanganan nilai-nilai outlier dengan teknik winsorize.
    - Menggunakan metode penskalaan seperti Standard Scaling untuk menormalkan fitur-fitur numerik.
    - Menerapkan teknik encoding seperti One-Hot Encoding dan Label Encoding untuk mengubah fitur-fitur kategorikal menjadi bentuk yang dapat dimengerti oleh model.
    - Menerapkan teknik undersampling untuk menyeimbangkan distribusi kolom target.
- Modeling:
    - Melatih data training dengan berbagai model machine learning termasuk Logistic Regression, Linear SVM, Decision Trees, Random Forest, dan XGBoost.
    - Mengevaluasi berbagai model machine learning menggunakan metrik evaluasi yang sesuai seperti akurasi, precision, recall, dan f1-score.

## Data Understanding
Data yang digunakan dalam proyek ini adalah data Bank Marketing yang dipublikasi oleh [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing). Data tersebut terkait dengan kampanye pemasaran langsung dari sebuah institusi perbankan Portugal. Kampanye pemasaran didasarkan pada panggilan telepon. Seringkali, lebih dari satu kontak ke klien yang sama diperlukan, untuk mengakses apakah produk (deposito berjangka bank) akan dilanggan ('yes') atau tidak ('no'). 

File yang akan digunakan adalah 'bank-full.csv' yang terdiri dari 45211 baris dengan target 'yes' sebanyak 5289 dan target 'no' sebanyak 39922 baris.

#### Variabel-variabel pada Bank Marketing UCI dataset adalah sebagai berikut:
Bank client data:
- age (numeric)
- job : tipe pekerjaan (categorical: "admin.",  "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services") 
- marital : status pernikahan (categorical: "married","divorced","single")
- education (categorical: "unknown","secondary","primary","tertiary")
- default: memiliki kredit yang gagal bayar? (binary: "yes","no")
- balance: saldo tahunan rata-rata, dalam euro (numeric) 
- housing: memiliki pinjaman perumahan? (binary: "yes","no")
-  loan: memiliki pinjaman pribadi? (binary: "yes","no")
terkait dengan kontak terakhir dari kampanye saat ini:
- contact: tipe komunikasi kontak (categorical: "unknown","telephone","cellular") 
- day: hari kontak terakhir dalam sebulan (numeric)
- month: bulan kontak terakhir dalam setahun (categorical: "jan", "feb", "mar", ..., "nov", "dec")
- duration: durasi kontak terakhir, dalam detik (numeric)

Atribut lain:
- campaign: jumlah kontak yang dilakukan selama kampanye ini dan untuk klien ini (numeric, termasuk kontak terakhir)
- pdays: jumlah hari yang telah berlalu setelah klien terakhir kali dihubungi dari kampanye sebelumnya (numeric -1 berarti klien tidak dihubungi sebelumnya)
- previous: jumlah kontak yang dilakukan sebelum kampanye ini dan untuk klien ini (numeric)
- poutcome: hasil dari kampanye pemasaran sebelumnya (categorical: "unknown", "other", "failure", "success")

Output variabel:
- y:  apakah klien telah berlangganan deposito berjangka? (binary: "yes","no")

#### Exploratory Data Analysis (EDA):
![eda bank marketing](https://github.com/ReginaAyumi/machine-learning-terapan/assets/90667044/9323a117-8a5c-49b6-b07c-2c6a29c98213)
![eda bank marketing (2)](https://github.com/ReginaAyumi/machine-learning-terapan/assets/90667044/6363c3a3-c75c-4575-bff2-217a9f97199c)
![eda bank marketing (1)](https://github.com/ReginaAyumi/machine-learning-terapan/assets/90667044/0d68fba8-8c13-42a3-b2cc-1e62ff417bf7)
Kategori pelanggan pada masing-masing fitur yang kemungkinan akan berlangganan deposit berjangka:
- Pekerjaan: management dan admin
- Status pernikahan : single
- Pendidikan: pendidikan tinggi
- Tipe komunikasi: seluler
- Tidak memiliki pinjaman rumah
- Tidak memiliki pinjaman pribadi
- Tidak memiliki kredit default
- Outcome dari kampanye marketing sebelumnya: sukses
- Bulan: Mei


## Data Preparation
1. Teknik Winsorize 
Teknik winsorize bertujuan untuk mengatasi outlier dalam dataset. Outlier merupakan nilai-nilai ekstrem yang dapat mengganggu analisis dan kinerja model machine learning. Dalam proses ini, setiap kolom numerik diproses secara terpisah, di mana nilai-nilai di luar rentang persentil 5 terendah dan persentil 95 tertinggi digantikan dengan nilai ambang batas tersebut. Dengan menggantikan nilai-nilai outlier, winsorization membantu meningkatkan kestabilan dan kinerja model machine learning.
    ```
    numeric_cols = df.select_dtypes(include=['int64']).columns
    for col in numeric_cols:
        df[col] = winsorize(df[col], limits=[0.05, 0.05])
    ```

2. Standard Scaling
Proses standarisasi fitur numerik dalam dataset, yang bertujuan untuk membuat distribusi nilai dari setiap fitur numerik memiliki mean 0 dan deviasi standar 1. Dalam langkah ini, digunakan objek StandardScaler dari library scikit-learn untuk melakukan standarisasi. Setiap fitur numerik diproses secara terpisah dan diubah sedemikian rupa sehingga nilai-nilainya memiliki distribusi normal standar.  Beberapa model machine learning seperti regresi linier dan SVM sensitif terhadap skala data. Dengan melakukan standar scaling, setiap fitur memiliki pengaruh yang seimbang pada model.
    ```
    from sklearn.preprocessing import StandardScaler
    
    numeric_cols = df.select_dtypes(include=['int64']).columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])
    ```
3. One Hot Encoding
Dalam tahapan ini, digunakan objek OneHotEncoder dari library scikit-learn. Data kategorikal dari kolom-kolom yang dipilih diubah menjadi representasi biner yang disimpan dalam variabel encoded_data. Banyak algoritma machine learning tidak dapat langsung menangani variabel kategorikal. Dengan menggunakan one hot encoding, informasi dari variabel kategorikal bisa dimasukkan ke dalam model tanpa menimbulkan bias.
    ```
    from sklearn.preprocessing import OneHotEncoder

    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[categorical_cols])
    ```
4. Label Encoding:
Label encoding digunakan untuk mengubah nilai-nilai kategorikal menjadi bilangan bulat.
Label encoding ini digunakan untuk mengkodekan variabel target 'y', dengan mengubah 'no' menjadi 0 dan 'yes' menjadi 1. Hal ini memungkinkan penggunaan algoritma machine learning yang memerlukan variabel target dalam bentuk numerik, seperti regresi logistik, decision trees dan random forests.
    ```
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    df['y'] = label_encoder.fit_transform(df['y'])
    ```
5. Teknik Undersampling
Dalam kasus ini, terdapat ketimpangan yang signifikan antara jumlah sampel dalam kelas mayoritas (majority class) dan jumlah sampel dalam kelas minoritas (minority class). Undersampling melibatkan penghapusan sampel-sampel dari kelas mayoritas sehingga proporsi antara kelas mayoritas dan kelas minoritas menjadi lebih seimbang. Hal ini dapat membantu model untuk belajar dengan lebih baik dari kelas minoritas tanpa terpengaruh oleh dominasi dari kelas mayoritas. Hasil dari undersampling ini adalah 5289 baris data 'yes' dan 'no'.
    ```
    from sklearn.utils import resample

    df_majority = df[df['y'] == 0]
    df_minority = df[df['y'] == 1]

    undersampled_majority = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
    df = pd.concat([df_minority, undersampled_majority])
    ```
6. Split Train dan Test Data
Split train dan test data dengan persentase 80% train dan 20% test data. Dipastikan juga bahwa data train dan data test dengan proporsi yang sama, yaitu 50% dengan menggunakan Stratified Shuffle Split.
    ```
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['y']), df['y'], test_size=0.2, random_state=42, stratify=df['y'])
    ```


## Modeling
Berikut adalah algoritma machine learning yang digunakan:
- **Logistic Regression** merupakan salah satu algoritma yang sederhana dan mudah diinterpretasi, cocok untuk tugas klasifikasi biner, serta tidak memerlukan asumsi distribusi normal pada fitur-fiturnya. Namun, algoritma ini kurang efektif dalam menangani masalah klasifikasi non-linear dan tidak dapat menangani interaksi antar fitur.
- **Linear SVM** efektif dalam menangani dataset dengan jumlah fitur yang besar dan dapat menangani dataset dengan dimensi tinggi. Algoritma ini juga cenderung tahan terhadap overfitting. Namun, Linear SVM sensitif terhadap skala fitur dan tidak efektif pada dataset dengan jumlah sampel yang sangat besar karena memerlukan waktu komputasi yang tinggi.
- **Decision Trees**  mampu menangani data numerik dan kategorikal tanpa memerlukan asumsi terhadap distribusi data. Namun, algoritma ini rentan terhadap overfitting dan tidak stabil, sehingga kecilnya perubahan pada data dapat menyebabkan perubahan yang signifikan pada struktur pohon.
- **Random Forest** mengatasi masalah overfitting yang umum terjadi pada decision trees dengan menggunakan teknik ensemble learning. Algoritma ini memerlukan lebih banyak sumber daya komputasi dibandingkan dengan decision trees tunggal dan tidak se-spesifik dalam interpretasi model seperti decision trees tunggal.
- **XGBoost (Extreme Gradient Boosting)** memiliki kinerja yang tinggi dalam berbagai tugas machine learning, terutama dalam dataset yang besar. Algoritma ini memiliki regularisasi yang efektif untuk mengurangi overfitting. Namun, XGBoost membutuhkan waktu komputasi yang lebih lama dibandingkan dengan beberapa algoritma lainnya karena iteratif.

##### Tahapan pemodelan machine learning
1. Import library model yang akan digunakan
    ```
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    import xgboost
    ```
2. Inisialisasi model dalam dictionary
    ```
    dict_classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Linear SVM": SVC(),
        "Decision Tree": tree.DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=18),
        "xgb": xgboost.XGBClassifier()
    }
    ```
     Setiap model diinisialisasi dengan parameter defaultnya, yang biasanya adalah parameter paling sederhana yang disediakan oleh implementasi algoritma tersebut.
3. Training model
    ```
    classifier.fit(X_train, y_train)
    ```
4. Prediksi akurasi
    ```
    classification_report(y_test, test_pred)
    ```

## Evaluation
Untuk menentukan kinerja model, perlu untuk mengevaluasi model yang sudah dibangun. Model klasifikasi akan dievaluasi menggunakan kriteria evaluasi seperti akurasi, presisi (precision), recall, dan f1-score. Persamaan-persamaan berikut menunjukkan perhitungan untuk mendapatkan evaluasi model.	

- Akurasi adalah rasio prediksi yang benar terhadap jumlah estimasi secara keseluruhan. Rumus untuk menghitung akurasi ditunjukkan dalam Persamaan berikut:
Akurasi=  (TP+TN)/(TP+TN+FP+FN)
- Presisi (Precision) merupakan perbandingan antara jumlah prediksi positif yang tepat dengan keseluruhan hasil prediksi positif. Presisi dihitung menggunakan persamaan berikut:
Precision=  TP/(TP+FP)	
- Recall adalah perbandingan antara jumlah prediksi positif dengan jumlah data positif secara keseluruhan. Recall dihitung menggunakan persamaan berikut:
Recall=  TP/(TP+FN)
- F1-score adalah suatu bentuk keseimbangan yang menggabungkan akurasi dan recall dalam sebuah sistem. Ini merupakan nilai rata-rata harmonis antara presisi dan recall. F1-score dihitung menggunakan persamaan berikut:
F1-score=  (2 x precision x recall)/(precision + recall) 

##### Hasil Evaluasi Model 

| Classifier         | Train Accuracy | Test Accuracy | Precision (Class 0) | Recall (Class 0) | F1-score (Class 0) | Precision (Class 1) | Recall (Class 1) | F1-score (Class 1) |
|--------------------|----------------|---------------|---------------------|------------------|---------------------|---------------------|------------------|---------------------|
| Logistic Regression| 0.8318         | 0.8417        | 0.84                | 0.84             | 0.84                | 0.84                | 0.85             | 0.84                |
| Linear SVM         | 0.8737         | 0.8620        | 0.89                | 0.83             | 0.86                | 0.84                | 0.90             | 0.87                |
| Decision Tree      | 1.0000         | 0.7944        | 0.80                | 0.79             | 0.79                | 0.79                | 0.80             | 0.80                |
| Random Forest      | 0.9970         | 0.8563        | 0.87                | 0.84             | 0.85                | 0.85                | 0.87             | 0.86                |
| XGBoost            | 0.9709         | 0.8691        | 0.89                | 0.85             | 0.87                | 0.85                | 0.89             | 0.87                |


Berdasarkan hasil evaluasi tersebut, berikut adalah kesimpulan yang didapat:

- Logistic Regression: Model ini memiliki akurasi yang cukup baik di atas 80%. Precision, recall, dan f1-score untuk kelas positif (1) dan negatif (0) juga cukup seimbang, menunjukkan kinerja yang stabil.
- Linear SVM: Model ini memberikan akurasi yang tinggi, sedikit lebih baik dari Logistic Regression. Precision, recall, dan f1-score untuk kelas positif (1) dan negatif (0) juga cukup seimbang, menunjukkan kinerja yang stabil.
- Decision Tree: Meskipun model Decision Tree memiliki akurasi yang cukup tinggi, namun terdapat sedikit overfitting karena akurasi pada data train mencapai 100%. Selain itu, precision, recall, dan f1-score untuk kedua kelas cenderung seimbang, menunjukkan bahwa model ini dapat melakukan klasifikasi dengan baik.
- Random Forest: Model Random Forest memberikan akurasi yang baik dan cenderung mengurangi overfitting yang terjadi pada Decision Tree karena penggunaan ensemble learning. Precision, recall, dan f1-score untuk kedua kelas juga cukup seimbang, menunjukkan kinerja yang baik.
- XGBoost: XGBoost merupakan model terbaik dari semua model yang dievaluasi. Model ini memberikan akurasi yang tinggi dan kinerja yang stabil dengan precision, recall, dan f1-score yang seimbang untuk kedua kelas.

Dengan demikian, berdasarkan hasil evaluasi, dapat disimpulkan bahwa XGBoost adalah model terbaik untuk tugas klasifikasi ini, diikuti oleh Linear SVM dan Random Forest. Sedangkan Decision Tree cenderung overfitting, dan Logistic Regression memiliki performa yang cukup baik tetapi lebih rendah dibandingkan dengan model lainnya.


### Referensi
[1] M. S. Başarslan and İ. D. Argun, "Classification Of a bank data set on various data mining platforms," 2018 Electric Electronics, Computer Science, Biomedical Engineerings' Meeting (EBBT), Istanbul, Turkey, 2018, pp. 1-4, doi: 10.1109/EBBT.2018.8391441. keywords: {Data mining;Classification algorithms;Decision trees;Data models;Training;Industries;Communications technology;data mining;banking;customer acquisition;data mining programs}







# Laporan Proyek Machine Learning - Regina Ayumi Ulayyaa

## Domain Proyek

Dalam industri perbankan, kampanye pemasaran langsung memiliki peran yang krusial dalam meningkatkan penjualan produk atau layanan seperti deposito berjangka, kartu kredit, atau pinjaman. Strategi pemasaran langsung ini melibatkan kontak langsung dengan calon pelanggan melalui panggilan telepon, email, atau surat pos. Namun, tantangan utama yang dihadapi oleh perusahaan dalam kampanye pemasaran langsung adalah bagaimana meningkatkan efektivitasnya agar lebih banyak pelanggan yang merespons dan berlangganan produk atau layanan yang ditawarkan.

Riset sebelumnya telah menunjukkan bahwa penggunaan teknik *machine learning*, terutama algoritma klasifikasi seperti *Decision Tree*, *K-Nearest Neighbor* (KNN), dan *Naive Bayes*, dapat membantu perusahaan dalam menganalisis pola-pola dalam data historis kampanye pemasaran langsung. Dengan memanfaatkan data historis ini, perusahaan dapat membangun model prediktif yang mampu memprediksi keputusan pelanggan, sehingga memungkinkan mereka untuk menargetkan calon pelanggan dengan lebih efisien dan meningkatkan tingkat kesuksesan kampanye mereka [1].

Proyek ini akan menggunakan data terkait kampanye pemasaran langsung dari sebuah lembaga perbankan di Portugal sebagai studi kasus. Data ini mencakup informasi tentang kontak-kontak yang dilakukan kepada calon pelanggan melalui panggilan telepon dan hasil dari kontak tersebut [2]. Dengan menerapkan teknik *machine learning*, seperti standarisasi data, *one-hot encoding*, dan algoritma klasifikasi seperti *Logistic Regression*, *Linear SVM*, *Decision Trees*, *Random Forest*, dan *XGBoost*, proyek ini bertujuan untuk membangun model prediktif yang menghasilkan akurasi terbaik yang dapat memprediksi keputusan pelanggan dalam berlangganan deposito berjangka berdasarkan data historis kampanye pemasaran langsung. Diharapkan hasil dari proyek ini dapat membantu perusahaan dalam meningkatkan efektivitas kampanye pemasaran langsung mereka dan meningkatkan penjualan produk atau layanan mereka secara keseluruhan.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, berikut ini batasan masalah yang dapat diselesaikan dengan proyek ini:
- Bagaimana cara melakukan pemrosesan data yang optimal untuk meningkatkan performa model prediktif dalam memprediksi keputusan pelanggan untuk berlangganan deposito berjangka?
- Bagaimana mengevaluasi berbagai model *machine learning* yang berbeda untuk menentukan model yang paling sesuai dengan data dan memberikan hasil prediksi dengan akurasi terbaik?

### Goals

- Memastikan data yang digunakan untuk melatih model prediktif diproses dengan baik dan siap untuk digunakan. Langkah-langkah pemrosesan data seperti penanganan *outlier*, penskalaan fitur numerik, dan *encoding* fitur kategorikal harus diterapkan secara efisien untuk memastikan data yang digunakan untuk melatih model prediktif siap digunakan dan tidak mengandung *noise* yang tidak perlu.
- Mengembangkan model *machine learning* yang dapat memprediksi keputusan pelanggan dengan akurasi yang tinggi. Model tersebut harus mampu mengenali pola-pola dalam data dan memberikan prediksi yang dapat diandalkan untuk membantu bank dalam meningkatkan efektivitas kampanye pemasaran dan meningkatkan pendapatan.

### Solution statements
- *Preprocessing Data*:
    - Melakukan identifikasi dan penanganan nilai-nilai *outlier* dengan teknik *winsorize*.
    - Menggunakan metode penskalaan seperti *Standard Scaling* untuk menormalkan fitur-fitur numerik.
    - Menerapkan teknik *encoding* seperti *One-Hot Encoding* dan *Label Encoding* untuk mengubah fitur-fitur kategorikal menjadi bentuk yang dapat dimengerti oleh model.
    - Menerapkan teknik *undersampling* untuk menyeimbangkan distribusi kolom target.
- *Modeling*:
    - Melatih data *training* dengan berbagai model *machine learning* termasuk *Logistic Regression*, *Linear SVM*, *Decision Trees*, *Random Forest*, dan *XGBoost*.
    - Mengevaluasi berbagai model *machine learning* menggunakan metrik evaluasi yang sesuai seperti akurasi, *precision*, *recall*, dan *f1-score*.

## Data Understanding
Data yang digunakan dalam proyek ini adalah data *Bank Marketing* yang dipublikasi oleh [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing). Data tersebut terkait dengan kampanye pemasaran langsung dari sebuah institusi perbankan Portugal. Kampanye pemasaran didasarkan pada panggilan telepon. Seringkali, lebih dari satu kontak ke klien yang sama diperlukan, untuk mengakses apakah produk (deposito berjangka bank) akan dilanggan ('yes') atau tidak ('no'). 

File yang akan digunakan adalah 'bank-full.csv' yang terdiri dari 45211 baris dengan target 'yes' sebanyak 5289 dan target 'no' sebanyak 39922 baris.

#### Variabel-variabel pada *Bank Marketing* UCI dataset adalah sebagai berikut:
*Bank client data*:
- *age (numeric)*
- *job* : tipe pekerjaan (*categorical*: "admin.",  "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services")
- *marital* : status pernikahan (*categorical*: "married","divorced","single")
- *education* (*categorical*: "unknown","secondary","primary","tertiary")
- *default*: memiliki kredit yang gagal bayar? (*binary*: "yes","no")
- *balance*: saldo tahunan rata-rata, dalam euro (*numeric*) 
- *housing*: memiliki pinjaman perumahan? (*binary*: "yes","no")
-  *loan*: memiliki pinjaman pribadi? (*binary*: "yes","no")
terkait dengan kontak terakhir dari kampanye saat ini:
- *contact*: tipe komunikasi kontak (*categorical*: "unknown","telephone","cellular") 
- *day*: hari kontak terakhir dalam sebulan (*numeric*)
- *month*: bulan kontak terakhir dalam setahun (*categorical*: "jan", "feb", "mar", ..., "nov", "dec")
- *duration*: durasi kontak terakhir, dalam detik (*numeric*)

Atribut lain:
- *campaig*n: jumlah kontak yang dilakukan selama kampanye ini dan untuk klien ini (*numeric*, termasuk kontak terakhir)
- *pdays*: jumlah hari yang telah berlalu setelah klien terakhir kali dihubungi dari kampanye sebelumnya (*numeric*, -1 berarti klien tidak dihubungi sebelumnya)
- *previous*: jumlah kontak yang dilakukan sebelum kampanye ini dan untuk klien ini (*numeric*)
- *poutcom*e: hasil dari kampanye pemasaran sebelumnya (*categorical*: "unknown", "other", "failure", "success")

*Output* variabel:
- y:  apakah klien telah berlangganan deposito berjangka? (*binary*: "yes","no")

#### *Exploratory Data Analysis* (EDA):
![eda bank marketing](https://github.com/ReginaAyumi/machine-learning-terapan/assets/90667044/9323a117-8a5c-49b6-b07c-2c6a29c98213)
<p align="center">
Gambar 1. Visualisasi fitur *job, marital status*, dan *education* dengan target
</p>

![eda bank marketing (2)](https://github.com/ReginaAyumi/machine-learning-terapan/assets/90667044/6363c3a3-c75c-4575-bff2-217a9f97199c)
<p align="center">
Gambar 2. Visualisasi fitur *loan, housing*, dan *default* dengan target
</p>

![eda bank marketing (1)](https://github.com/ReginaAyumi/machine-learning-terapan/assets/90667044/0d68fba8-8c13-42a3-b2cc-1e62ff417bf7)
<p align="center">
Gambar 3. Visualisasi fitur *contact, poutcome*, dan *month* dengan kolom target
</p>


Berikut kategori pelanggan pada masing-masing fitur yang kemungkinan akan berlangganan deposit berjangka:
- Pekerjaan: *management* dan *admin*

Manajemen dan Admin adalah dua kategori pekerjaan yang paling cenderung untuk berlangganan deposito berjangka. Dapat dilihat pada Gambar 1 bahwa persentase "yes" untuk kedua kategori ini lebih tinggi daripada persentase "no" dibanding kategori lainnya. Hal ini dapat diinterpretasikan bahwa orang-orang dengan jabatan manajerial atau administratif mungkin memiliki stabilitas keuangan dan pengetahuan yang cukup untuk mempertimbangkan investasi jangka panjang seperti deposito berjangka. 
- Status pernikahan : *single*

Status pernikahan "*single*" juga cenderung untuk berlangganan deposito berjangka. Pada 
Gambar 1, perbandingan antara persentase "yes" lebih tinggi daripada "no" untuk kategori *single*. Ini mungkin disebabkan oleh kemandirian finansial yang lebih tinggi dan kemampuan untuk melakukan investasi secara mandiri tanpa pertimbangan pasangan.
- Pendidikan: pendidikan tinggi

Pelanggan dengan pendidikan tinggi lebih cenderung untuk berlangganan deposito berjangka. Ini dapat dijelaskan dengan asumsi bahwa orang dengan pendidikan lebih tinggi mungkin memiliki pemahaman yang lebih baik tentang manfaat investasi jangka panjang dan lebih siap secara finansial untuk melakukan investasi tersebut. Dapat dilihat pada Gambar 1 bahwa persentase "yes" untuk kategori ini lebih tinggi daripada persentase "no" dibanding kategori lainnya.
- Tidak memiliki pinjaman pribadi,tidak memiliki pinjaman rumah

Dapat dilihat pada Gambar 2 bahwa pelanggan yang tidak memiliki pinjaman rumah dan pinjaman pribadi cenderung untuk berlangganan deposito berjangka. Ini bisa menunjukkan bahwa mereka memiliki beban finansial yang lebih rendah dan lebih siap untuk melakukan investasi jangka panjang.
- Tidak memiliki kredit *default*

Dapat dilihat pada Gambar 2 bahwa pelanggan yang tidak memiliki catatan kredit *default* lebih cenderung untuk berlangganan deposito berjangka. Ini mungkin menandakan bahwa pelanggan tersebut memiliki riwayat kredit yang baik dan kemungkinan besar memiliki kestabilan keuangan yang cukup untuk melakukan investasi jangka panjang.
- Tipe komunikasi: seluler

Penggunaan komunikasi melalui seluler (telepon seluler) lebih cenderung menghasilkan pelanggan yang berlangganan deposito berjangka. Hal ini mungkin menunjukkan bahwa pelanggan yang menggunakan telepon seluler lebih aktif atau terhubung secara *digital*, yang juga dapat mencerminkan kesiapan untuk melakukan transaksi finansial secara elektronik, termasuk berlangganan deposito berjangka. Dapat dilihat pada Gambar 3 bahwa persentase "yes" untuk kategori ini lebih tinggi daripada persentase "no" dibanding kategori lainnya.
- *Outcome* dari kampanye marketing sebelumnya: sukses

Hasil positif dari kampanye pemasaran sebelumnya juga mempengaruhi kecenderungan pelanggan untuk berlangganan deposito berjangka. Ini menunjukkan bahwa pelanggan yang merespons positif terhadap kampanye pemasaran sebelumnya lebih mungkin untuk melakukan langkah selanjutnya dalam hubungan dengan perusahaan, seperti berlangganan deposito berjangka. Ini dapat dilihat dari perbandingan target "yes" dan "no" dalam Gambar 3.
- Bulan: Mei

Bulan Mei menunjukkan kecenderungan yang lebih tinggi untuk berlangganan deposito berjangka. Hal ini mungkin terkait dengan faktor musiman atau keadaan ekonomi yang khusus pada bulan tersebut yang mendorong orang untuk melakukan investasi jangka panjang. Ini dapat dilihat dari banyaknya orang yang mendaftar deposito berjangka pada Bulan Mei dalam Gambar 3.


## Data Preparation
1. Teknik *Winsorize*

Teknik *winsorize* bertujuan untuk mengatasi *outlier* dalam dataset. *Outlier* merupakan nilai-nilai ekstrem yang dapat mengganggu analisis dan kinerja model *machine learning*. Dalam proses ini, setiap kolom numerik diproses secara terpisah, di mana nilai-nilai di luar rentang persentil 5 terendah dan persentil 95 tertinggi digantikan dengan nilai ambang batas tersebut. Dengan menggantikan nilai-nilai *outlier*, *winsorization* membantu meningkatkan kestabilan dan kinerja model *machine learning*.

2. *Standard Scaling*

Proses standarisasi fitur numerik dalam dataset, yang bertujuan untuk membuat distribusi nilai dari setiap fitur numerik memiliki *mean* 0 dan standar deviasi 1. Dalam langkah ini, digunakan objek *StandardScaler* dari *library scikit-learn* untuk melakukan standarisasi. Setiap fitur numerik diproses secara terpisah dan diubah sedemikian rupa sehingga nilai-nilainya memiliki distribusi normal standar. Beberapa model *machine learning* seperti *logistic regression* dan SVM sensitif terhadap skala data. Dengan melakukan *standard scaling*, setiap fitur memiliki pengaruh yang seimbang pada model.

3. *One Hot Encoding*

Dalam tahapan ini, digunakan objek *OneHotEncoder* dari *library scikit-learn*. Data kategorikal dari kolom-kolom yang dipilih diubah menjadi representasi biner yang disimpan dalam variabel *encoded_data*. Banyak algoritma *machine learning* tidak dapat langsung menangani variabel kategorikal. Dengan menggunakan *one hot encoding*, informasi dari variabel kategorikal bisa dimasukkan ke dalam model tanpa menimbulkan bias.

4. *Label Encoding*:

*Label encoding* digunakan untuk mengubah nilai-nilai kategorikal menjadi bilangan bulat.
*Label encoding* ini digunakan untuk mengkodekan variabel target 'y', dengan mengubah 'no' menjadi 0 dan 'yes' menjadi 1. Hal ini memungkinkan penggunaan algoritma *machine learning* yang memerlukan variabel target dalam bentuk numerik, seperti *logistic regression, decision trees* dan *random forests*.

5. Teknik *Undersampling*

Dalam kasus ini, terdapat ketimpangan yang signifikan antara jumlah sampel dalam kelas mayoritas dan jumlah sampel dalam kelas minoritas. *Undersampling* melibatkan penghapusan sampel-sampel dari kelas mayoritas sehingga proporsi antara kelas mayoritas dan kelas minoritas menjadi lebih seimbang. Hal ini dapat membantu model untuk belajar dengan lebih baik dari kelas minoritas tanpa terpengaruh oleh dominasi dari kelas mayoritas. Hasil dari *undersampling* ini adalah 5289 baris data 'yes' dan 'no'.

6. *Split Train* dan *Test Data*

*Split train* dan *test data* dengan persentase 80% *train* dan 20% *test data*. Dipastikan juga bahwa data *train* dan data *test* dengan proporsi yang sama, yaitu 50% dengan menggunakan *Stratified Shuffle Split*.


## Modeling
Berikut adalah algoritma *machine learning* yang digunakan:
- ***Logistic Regression*** merupakana salah satu algoritma yang sederhana dan mudah diinterpretasi. Cocok untuk tugas klasifikasi biner dan tidak memerlukan asumsi distribusi normal pada fitur-fiturnya. Namun, kurang efektif dalam menangani masalah klasifikasi non-linear dan tidak dapat menangani interaksi antar fitur. *Logistic regression* cocok untuk klasifikasi berlangganan deposito berjangka karena kemampuannya dalam menghasilkan probabilitas prediksi, yang dapat diinterpretasikan secara langsung sebagai kemungkinan berlangganan.
- ***Linear SVM*** efektif dalam menangani dataset dengan jumlah fitur yang besar dan dapat menangani dataset dengan dimensi tinggi. Algoritma ini juga cenderung tahan terhadap *overfitting*. Namun, *Linear SVM* sensitif terhadap skala fitur dan tidak efektif pada dataset dengan jumlah sampel yang sangat besar karena memerlukan waktu komputasi yang tinggi. *Dataset* yang digunakan adalah klasifikasi biner (berlangganan atau tidak berlangganan) sehingga SVM dapat menangani dataset dengan dimensi tinggi dan relatif kecil dengan baik.
- ***Decision Trees***  mampu menangani data numerik dan kategorikal tanpa memerlukan asumsi terhadap distribusi data. Namun, algoritma ini rentan terhadap *overfitting* dan tidak stabil, sehingga kecilnya perubahan pada data dapat menyebabkan perubahan yang signifikan pada struktur pohon. *Decision Trees* dipilih karena kemampuannya untuk mengatasi masalah klasifikasi biner serta kemampuan untuk membangun model yang mudah diinterpretasikan.
- ***Random Forest*** mengatasi masalah *overfitting* yang umum terjadi pada *decision trees* dengan menggunakan teknik *ensemble learning*. Algoritma ini memerlukan lebih banyak sumber daya komputasi dibandingkan dengan *decision trees* tunggal dan tidak se-spesifik dalam interpretasi model seperti *decision trees* tunggal. *Random Forest* dipilih karena kemampuannya untuk menangani *overfitting* dan menghasilkan prediksi yang lebih stabil.
- ***XGBoost (Extreme Gradient Boosting)*** memiliki kinerja yang tinggi dalam berbagai tugas *machine learning*, terutama dalam *dataset* yang besar. Algoritma ini memiliki regularisasi yang efektif untuk mengurangi *overfitting*. Namun, *XGBoost* membutuhkan waktu komputasi yang lebih lama dibandingkan dengan beberapa algoritma lainnya karena iteratif. *XGBoost* dipilih karena kinerja yang tinggi dan kemampuannya untuk menangani data dengan jumlah fitur yang besar.

#### Tahapan pemodelan *machine learning*
1. *Import library* model yang akan digunakan:

Tahap pertama adalah mengimpor *library* atau modul yang menyediakan implementasi dari model *machine learning* yang akan digunakan dalam proyek. *Library* yang digunakan yaitu *pandas* untuk mengolah data dan *Scikit-learn (sklearn)* untuk model-model klasifikasi seperti *Logistic Regression*, *Linear SVM*, *Decision Trees*, *Random Forest*, dan *XGBoost*.

2. Import dataset

Data *Bank Marketing* diunduh dari *website* [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) dan dimasukkan ke dalam *DataFrame Pandas*.

3. Inisialisasi model dalam *dictionary*

Langkah selanjutnya adalah menginisialisasi setiap model dalam sebuah *dictionary*. Setiap model diinisialisasi dengan parameter *default*nya. Dalam kasus ini, *dictionary* bernama *dict_classifiers* digunakan untuk menyimpan model-model tersebut. Parameter *default* biasanya dipilih karena merupakan parameter paling sederhana yang disediakan oleh implementasi algoritma tersebut.

4. *Training model*

Setelah model diinisialisasi, langkah selanjutnya adalah melatih (*train*) model menggunakan data *training*. Data *training* adalah *dataset* yang digunakan untuk mengajarkan model agar dapat mempelajari pola-pola atau hubungan dalam data. Proses pelatihan ini dilakukan dengan memanggil fungsi atau metode *fit*() pada setiap model, dengan menggunakan fitur-fitur dan label dari data *training*.

5. Prediksi akurasi

Setelah model dilatih, langkah terakhir adalah melakukan prediksi pada data uji (*test data*) dan mengevaluasi akurasinya. Evaluasi ini dilakukan dengan menggunakan metrik-metrik seperti akurasi, *precision*, *recall*, dan *f1-score*. Prediksi pada data uji dilakukan dengan memanggil fungsi atau metode *predict*() pada setiap model, dengan menggunakan fitur-fitur dari data uji. Setelah mendapatkan prediksi, hasilnya dibandingkan dengan label sebenarnya pada data uji untuk menghitung metrik evaluasi yang sesuai.

## Evaluation
Untuk menentukan kinerja model, perlu untuk mengevaluasi model yang sudah dibangun. Model klasifikasi akan dievaluasi menggunakan kriteria evaluasi seperti akurasi, presisi (*precision*), *recall*, dan *f1-score*. Persamaan-persamaan berikut menunjukkan perhitungan untuk mendapatkan evaluasi model.	

- Akurasi adalah rasio prediksi yang benar terhadap jumlah estimasi secara keseluruhan. Rumus untuk menghitung akurasi ditunjukkan dalam Persamaan berikut:

  $$ Accuracy = {TP+TN \over(TP+TN+FP+FN)} $$
- Presisi (*Precision*) merupakan perbandingan antara jumlah prediksi positif yang tepat dengan keseluruhan hasil prediksi positif. Presisi dihitung menggunakan persamaan berikut:

$$ Precision = {TP \over(TP+FP)} $$
- *Recall* adalah perbandingan antara jumlah prediksi positif dengan jumlah data positif secara keseluruhan. *Recall* dihitung menggunakan persamaan berikut:

$$ Recall = {TP \over(TP+FN)} $$
- *F1-score* adalah suatu bentuk keseimbangan yang menggabungkan akurasi dan *recall* dalam sebuah sistem. Ini merupakan nilai rata-rata harmonis antara presisi dan *recall*. *F1-score* dihitung menggunakan persamaan berikut:

$$ F1-score = {2* precision*recall \over precision+ recall} $$


#### Hasil Evaluasi Model 

Tabel 1. Hasil evaluasi model *machine learning*

| *Classifier*        | *Train Accuracy* | *Test Accuracy* | *Precision (Class 0)* | *Recall (Class 0)* | *F1-score (Class 0)* | *Precision (Class 1)* | *Recall (Class 1)* | *F1-score (Class 1)* |
|--------------------|----------------|---------------|---------------------|------------------|---------------------|---------------------|------------------|---------------------|
| *Logistic Regression*| 0.8318         | 0.8417        | 0.84                | 0.84             | 0.84                | 0.84                | 0.85             | 0.84                |
| *Linear SVM*       | 0.8737         | 0.8620        | 0.89                | 0.83             | 0.86                | 0.84                | 0.90             | 0.87                |
| *Decision Tree*     | 1.0000         | 0.7944        | 0.80                | 0.79             | 0.79                | 0.79                | 0.80             | 0.80                |
| *Random Forest*      | 0.9970         | 0.8563        | 0.87                | 0.84             | 0.85                | 0.85                | 0.87             | 0.86                |
| *XGBoost*            | 0.9709         | 0.8691        | 0.89                | 0.85             | 0.87                | 0.85                | 0.89             | 0.87                |


Berdasarkan hasil evaluasi model pada Tabel 1, berikut adalah kesimpulan yang didapat:

- *Logistic Regression*: Model ini memiliki akurasi yang cukup baik di atas 80%. *Precision, recall,* dan *f1-score* untuk kelas positif (1) dan negatif (0) juga cukup seimbang, menunjukkan kinerja yang stabil.
- *Linear SVM*: Model ini memberikan akurasi yang tinggi, sedikit lebih baik dari *Logistic Regression*. *Precision, recall*, dan *f1-score* untuk kelas positif (1) dan negatif (0) juga cukup seimbang, menunjukkan kinerja yang stabil.
- *Decision Tree*: Meskipun model *Decision Tree* memiliki akurasi yang cukup tinggi, namun terdapat sedikit *overfitting* karena akurasi pada data *train* mencapai 100%. Selain itu, *precision, recall*, dan *f1-score* untuk kedua kelas cenderung seimbang, menunjukkan bahwa model ini dapat melakukan klasifikasi dengan baik.
- *Random Forest*: Model *Random Forest* memberikan akurasi yang baik dan cenderung mengurangi *overfitting* yang terjadi pada *Decision Tree* karena penggunaan *ensemble learning*. *Precision, recall*, dan *f1-score* untuk kedua kelas juga cukup seimbang, menunjukkan kinerja yang baik.
- *XGBoost*: *XGBoost* merupakan model terbaik dari semua model yang dievaluasi. Model ini memberikan akurasi yang tinggi dan kinerja yang stabil dengan *precision, recall, dan f1-score* yang seimbang untuk kedua kelas.

Dari hasil evaluasi model, terlihat bahwa semua model memiliki tingkat akurasi yang cukup baik, dengan beberapa model seperti *Linear SVM, Random Forest*, dan *XGBoost* mencapai akurasi di atas 85%. Namun, dalam mengevaluasi kinerja model, perlu juga dipertimbangkan presisi, *recall*, dan *f1-score*, karena data ini merupakan data yang berkaitan dengan pemasaran langsung kepada pelanggan.

Dalam hal ini, *XGBoost* menonjol sebagai model terbaik dengan kinerja yang stabil pada semua metrik evaluasi. Dengan hasil evaluasi ini, dapat disimpulkan bahwa proyek ini berhasil mencapai tujuannya untuk membangun model prediktif dengan tingkat akurasi yang tinggi. Dengan demikian, model ini dapat membantu bank dalam mengidentifikasi calon pelanggan yang potensial untuk berlangganan deposito berjangka, sehingga meningkatkan efektivitas kampanye pemasaran mereka.

### Referensi
[1] M. S. Başarslan and İ. D. Argun, "Classification Of a bank data set on various data mining platforms," 2018 Electric Electronics, Computer Science, Biomedical Engineerings' Meeting (EBBT), Istanbul, Turkey, 2018, pp. 1-4, doi: 10.1109/EBBT.2018.8391441. keywords: {Data mining;Classification algorithms;Decision trees;Data models;Training;Industries;Communications technology;data mining;banking;customer acquisition;data mining programs}

[2] UCI Machine Learning Repository. (2012). Bank Marketing Data Set. Diakses dari https://archive.ics.uci.edu/dataset/222/bank+marketing

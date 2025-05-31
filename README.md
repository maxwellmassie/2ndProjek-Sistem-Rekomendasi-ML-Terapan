# Laporan Proyek Machine Learning - Gold Stein Maxwell Massie
## Sistem Rekomendasi Anime

## Project Overview

### Latar Belakang
Pertumbuhan pesat industri hiburan digital, khususnya anime, menyebabkan kebutuhan akan sistem rekomendasi yang mampu membantu pengguna menemukan konten sesuai preferensi mereka. Dengan ribuan judul anime dan beragam selera pengguna, sistem rekomendasi menjadi solusi penting untuk meningkatkan pengalaman pengguna.

Metode Content-Based Filtering yang memanfaatkan fitur konten seperti genre dan Collaborative Filtering yang menggunakan interaksi pengguna terbukti efektif dalam merekomendasikan item yang relevan. Penggabungan kedua metode ini membantu mengatasi keterbatasan masing-masing dan memberikan rekomendasi yang lebih akurat.

Menurut Pazzani dan Billsus (2018), Content-Based Recommendation Systems adalah sistem yang merekomendasikan item kepada pengguna berdasarkan deskripsi item dan profil minat pengguna. Sistem ini dapat digunakan di berbagai domain seperti rekomendasi halaman web, artikel berita, restoran, program televisi, dan barang dagangan [1]. Di Indonesia, penelitian oleh Faurina dan Sitanggang (2023) menerapkan Content-Based Filtering dan Collaborative Filtering pada sistem rekomendasi destinasi wisata di Bali. Penelitian ini menunjukkan bahwa pendekatan hybrid tersebut efektif dalam memberikan rekomendasi personal dengan performa evaluasi yang baik [2].

Proyek ini bertujuan membangun sistem rekomendasi anime menggunakan dataset MyAnimeList, memanfaatkan teknik Content-Based Filtering dan Collaborative Filtering untuk memberikan rekomendasi yang personal dan relevan.

**Referensi**

[1] M. J. Pazzani dan D. Billsus, “Content-based recommendation systems,” dalam The Adaptive Web, P. Brusilovsky, A. Kobsa, dan W. Nejdl, Eds. Berlin, Heidelberg: Springer, 2018, hlm. 325–341. Tersedia online: https://doi.org/10.1007/978-3-540-72079-9_10

[2] R. Faurina dan E. Sitanggang, “Implementasi Metode Content-Based Filtering dan Collaborative Filtering pada Sistem Rekomendasi Wisata di Bali,” Techno.COM, vol. 22, no. 4, pp. 870-881, Nov. 2023.
Tersedia online: https://core.ac.uk/download/pdf/595469606.pdf


## Business Understanding

Dalam era digital saat ini, pengguna anime menghadapi tantangan dalam memilih judul yang sesuai dari ribuan pilihan yang tersedia. Sistem rekomendasi yang efektif sangat dibutuhkan untuk membantu pengguna menemukan anime yang sesuai dengan preferensi mereka berdasarkan interaksi dan karakteristik konten.

### Problem Statements
1. Bagaimana membantu pengguna menemukan anime yang sesuai dengan preferensi mereka berdasarkan data rating historis dan fitur genre anime?
2. Bagaimana membangun model rekomendasi yang akurat dan efisien dengan memanfaatkan data interaksi pengguna serta informasi konten anime untuk menghasilkan rekomendasi yang personal dan relevan?

### Goals
1. Mengembangkan sistem rekomendasi anime yang mampu memberikan rekomendasi personal dengan memadukan preferensi pengguna dari data rating dan karakteristik konten anime, terutama genre.
2. Memanfaatkan data interaksi pengguna dan fitur genre anime untuk menghasilkan rekomendasi yang akurat dan sesuai, dengan pendekatan pengolahan data interaksi (rating) dan analisis kesamaan konten (genre).
3. Membangun dan mengimplementasikan model rekomendasi berbasis Content-Based Filtering dan Collaborative Filtering menggunakan Matrix Factorization, guna meningkatkan kualitas dan akurasi rekomendasi anime yang diberikan kepada pengguna.

### Solution Statement
1. Content-Based Filtering
Menggunakan TF-IDF Vectorizer pada fitur genre dan deskripsi anime untuk menghasilkan representasi konten, kemudian menggunakan Cosine Similarity untuk merekomendasikan anime dengan konten mirip yang disukai pengguna.

2. Collaborative Filtering dengan Matrix Factorization menggunakan Embedding Layer di TensorFlow/Keras
Membuat model deep learning untuk mempelajari representasi laten (embedding) pengguna dan anime berdasarkan data rating, sehingga dapat memprediksi rating dan merekomendasikan anime yang sesuai dengan preferensi pengguna.

### Evaluasi Keberhasilan
Keberhasilan proyek ini akan diukur menggunakan recomender system precision, Mean Squared Error, dan Root Mean Squared Error
1. Presisi adalah metrik yang digunakan untuk mengukur seberapa relevan item yang direkomendasikan oleh sistem. Ini dihitung sebagai rasio jumlah rekomendasi yang relevan dengan total jumlah item yang direkomendasikan.
2. Dalam sistem rekomendasi berbasis Collaborative Filtering, Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE) adalah metrik umum yang digunakan untuk mengevaluasi akurasi prediksi model. Keduanya mengukur rata-rata kuadrat atau akar kuadrat dari kesalahan (selisih antara nilai prediksi dan nilai sebenarnya).


## Data Understanding
Dataset ini digunakan untuk membangun sistem rekomendasi anime berbasis Collaborative Filtering dengan pendekatan Matrix Factorization. Data ini mencakup informasi rating dari 73.516 pengguna terhadap 12.294 anime, serta metadata dari setiap anime. Variabel-variabel yang tersedia meliputi judul anime, genre, jenis (movie, TV, OVA, dll), jumlah episode, rata-rata rating, dan jumlah anggota komunitas. Sementara itu, dataset rating mencatat interaksi pengguna berupa rating yang diberikan terhadap anime tertentu, termasuk nilai -1 untuk anime yang ditonton tetapi tidak diberi rating.

Dataset ini sangat cocok untuk eksplorasi data, pembangunan model rekomendasi berbasis pembelajaran mesin, serta pemahaman pola preferensi pengguna terhadap berbagai genre dan jenis anime.
- Anime User Rating and Metadata for Recommendation System: [Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database/).

### Variabel-variabel pada Health and Lifestyle Data for Regression dataset adalah sebagai berikut:
#### 📁 `Anime.csv` (12.294 data anime)
![image](https://github.com/user-attachments/assets/d675792f-f4be-49c8-984c-74ce47226f53)

* **anime_id** (`int64`): ID unik untuk setiap anime dari situs MyAnimeList. Digunakan sebagai penghubung dengan data rating.
* **name** (`object`): Nama lengkap atau judul dari anime.
* **genre** (`object`): Daftar genre dari anime, dipisahkan dengan koma (misalnya: Action, Adventure, Drama, Fantasy, Magic,).
* **type** (`object`): Tipe tayangan dari anime (misalnya TV, Movie, OVA, dll).
* **episodes** (`object`): Jumlah episode dari anime. Disimpan sebagai teks karena beberapa nilai bisa berupa 'Unknown' atau tidak terisi secara numerik.
* **rating** (`float64`): Rata-rata rating dari pengguna terhadap anime tersebut dalam skala 1–10.
* **members** (`int64`): Jumlah anggota komunitas yang memasukkan anime ke dalam daftar mereka. Mewakili tingkat popularitas anime.

#### 📁 `Rating.csv` (7.813.737 data interaksi pengguna)
![image](https://github.com/user-attachments/assets/77abc7e5-98ec-4e0d-a658-e17651d9fb5d)

* **user_id** (`int64`): ID anonim dari pengguna. Dibuat secara acak dan tidak mengandung informasi identitas.
* **anime_id** (`int64`): ID anime yang dirating oleh pengguna. Berfungsi sebagai *foreign key* untuk menghubungkan ke `anime_id` di `Anime.csv`.
* **rating** (`int64`): Rating yang diberikan pengguna terhadap anime. Skala 1–10. Jika bernilai -1, berarti pengguna telah menonton anime tersebut tetapi tidak memberikan rating eksplisit.


### Exploratory Data Analysis (EDA)
#### Menampilkan info data Anime.csv
| Kolom      | Non-Null Count | Tipe Data |
| :--------- | :------------- | :-------- |
| `anime_id` | 12294          | `int64`   |
| `name`     | 12294          | `object`  |
| `genre`    | 12232          | `object`  |
| `type`     | 12269          | `object`  |
| `episodes` | 12294          | `object`  |
| `rating`   | 12064          | `float64` |
| `members`  | 12294          | `int64`   |

**Ringkasan Tipe Data:**
* `float64`: 1 kolom
* `int64`: 2 kolom
* `object`: 4 kolom

**Total Memori:** 672.5+ KB

#### Menampilkan info data Rating.csv
| Kolom     | Tipe Data |
| :-------- | :-------- |
| `user_id` | `int64`   |
| `anime_id`| `int64`   |
| `rating`  | `int64`   |

**Ringkasan Tipe Data:**
* `int64`: 3 kolom

**Total Memori:** 178.8 MB

#### Menampilkan Statistik Deskriptif Anime.csv
![image](https://github.com/user-attachments/assets/e26dfd07-9a2a-4a5a-9c9c-adb0278c22d9)

Dataset anime memiliki 12.017 data lengkap untuk kolom rating dan members. Rata-rata rating anime sekitar 6,48 dengan variasi yang cukup kecil (std 1,02), nilai rating terendah 1,67 dan tertinggi sempurna 10. Jumlah anggota komunitas (members) bervariasi sangat luas, dengan rata-rata sekitar 18.349, dari minimum 12 hingga lebih dari 1 juta, menunjukkan adanya anime yang sangat populer dan yang kurang dikenal. Sebaran jumlah anggota ini sangat besar dengan standar deviasi tinggi (55.372), menandakan variasi signifikan dalam popularitas anime.


#### Menampilkan Statistik Deskriptif Rating.csv
![image](https://github.com/user-attachments/assets/0cd3e1bd-a7a9-4be5-aae5-be50a490d5f1)

Statistik deskriptif untuk kolom rating pada dataset interaksi pengguna menunjukkan terdapat 7.813.737 data dengan rata-rata rating sekitar 6,14 dari skala 1 hingga 10. Rating memiliki variasi yang cukup besar dengan standar deviasi 3,73, nilai minimum -1 (menandakan anime ditonton tanpa rating), dan nilai maksimum 10. Sebagian besar rating berada di kisaran 6 hingga 9, dengan median di angka 7, menunjukkan preferensi pengguna cenderung positif.


#### Mengecek & Menampilkan Missing Value
![image](https://github.com/user-attachments/assets/299e6343-bed7-48b8-a2c0-34a7802dacc3)

Pada dataset anime, terdapat beberapa nilai yang hilang, yaitu 62 pada kolom genre, 25 pada kolom type, dan 230 pada kolom rating. Sementara itu, dataset rating tidak memiliki nilai yang hilang sama sekali, semua kolom user_id, anime_id, dan rating lengkap tanpa missing value.

#### Mengecek & Menampilkan Data Duplicate
![image](https://github.com/user-attachments/assets/42d94c8b-0df6-4fc2-98e2-59c3b0748cbe)

Dataset anime tidak mengandung data duplikat sama sekali, sedangkan pada dataset rating ditemukan 1 baris duplikat. Dengan jumlah data yang sangat besar, keberadaan satu baris duplikat pada rating ini relatif kecil dan dapat dihapus untuk menjaga kualitas data.


#### Visualisasi Top 10 Genre Anime
![image](https://github.com/user-attachments/assets/d673475f-897e-402c-a7f2-73011002778e)

Berdasarkan diagram batang "Top 10 Genre Anime", genre "Comedy" mendominasi dengan jumlah tertinggi (4645), diikuti oleh "Action" (2845) dan "Adventure" (2348). Selanjutnya, "Fantasy" dan "Sci-Fi" memiliki frekuensi yang mirip, dan secara bertahap menurun hingga genre "School" yang paling jarang muncul dalam daftar 10 teratas ini.


#### Visualisasi Berdasarkan type Anime
![image](https://github.com/user-attachments/assets/a2711f91-0183-495a-8aca-15baaa73312f)

Berdasarkan diagram batang "Jumlah Anime berdasarkan Tipe", tipe anime "TV" merupakan yang paling banyak dengan total 3787, diikuti oleh "OVA" (3311) dan "Movie" (2348). Tipe "Special" memiliki jumlah yang lebih rendah (1676), dan kemudian secara signifikan menurun untuk tipe "ONA" (659) dan "Music" (488) yang merupakan jumlah terkecil dalam kategori ini.



## Data Preprocessing & Preparation
1. **Sampling Data Rating**: Karena dataset rating sangat besar(7.813.737 data), maka mengambil sampel 50.000 baris saja secara acak untuk mempercepat proses dan mengurangi beban komputasi, sambil tetap menjaga representasi data.
```python
# 1. Sampling Data Rating
df_rating = df_rating.sample(n=50000, random_state=42).reset_index(drop=True)

# mengecek info setelah sampling
print(df_rating.info())

# mengecek beberapa baris awal
print(df_rating.head())
```

**output**:

![image](https://github.com/user-attachments/assets/40eb8f95-2952-4a57-8140-e552a8cdcac8)

Hasilnya mereplace DataFrame df_rating baru yang berisi 50.000 baris acak. DataFrame ini memiliki 3 kolom (user_id, anime_id, dan rating), semuanya bertipe integer (int64), dan tidak ada nilai kosong di ketiga kolom tersebut. Ini menunjukkan bahwa proses sampling berjalan sesuai harapan dan menghasilkan subset data yang bersih untuk analisis lebih lanjut.



2. **Membersihkan Nama Anime**: Menghapus karakter khusus pada kolom nama agar data teks lebih bersih dan konsisten, memudahkan proses analisis dan pencocokan nama.
```python
# 2. Membersihkan Nama Anime dengan mengapus karakter khusus pada kolom nama
df_anime['name'] = df_anime['name'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
```

3. **Handling Missing Values**: Menghapus baris yang memiliki nilai kosong di kolom penting seperti genre, tipe, dan rating agar model dan analisis tidak terganggu oleh data yang tidak lengkap atau bias data.
```python
# 3. Handling Missing Values
# Menghapus baris yang memiliki missing value di kolom 'genre', 'type', dan 'rating'
df_anime.dropna(subset=['genre', 'type', 'rating'], inplace=True)

# Mengecek kembali Missing Values
print("Missing Values in df_anime:")
print(df_anime.isnull().sum())

print("\nMissing Values in df_rating:")
print(df_rating.isnull().sum())
```

**output**:

![image](https://github.com/user-attachments/assets/35d2a6e4-b895-41ea-804f-103412a80ff7)

Setelah tahap pembersihan data, hasil pengecekan menunjukkan bahwa kedua dataset, df_anime dan df_rating, kini bebas dari nilai yang hilang (missing values) di semua kolomnya, siap untuk analisis dan pemodelan lebih lanjut.

4. **Menghapus Data Duplikat**: Menghilangkan baris duplikat pada data rating untuk menjaga integritas data dan menghindari bias berlebih dalam model rekomendasi.
```python
# 4. Menghapus Data Duplikat
df_rating.drop_duplicates(inplace=True)

print("\nDuplicate Rows in df_rating:")
print(df_rating.duplicated().sum())
```

**output**:

![image](https://github.com/user-attachments/assets/1b69b090-4540-46db-81b2-9bb3b9730660)

Setelah tahap pembersihan data, hasil pengecekan menunjukkan bahwa tidak ada duplikat data pada df_rating.


5. **Standarisasi Genre**: Mengambil genre pertama dari setiap anime dan menyamakan genre berdasarkan nama anime dengan mengambil genre yang paling sering muncul agar genre menjadi konsisten dan menghindari duplikasi atau inkonsistensi pada fitur genre.
```python
# 5. Standarisasi Genre
# mengambil genre pertama dari setiap data
df_anime['genre'] = df_anime['genre'].apply(lambda x: x.split(',')[0].strip())

# Menyamakan Jenis genre berdsarkan name
# Ambil genre yang paling umum untuk setiap nama
genre_consensus = (
    df_anime.groupby('name')['genre']
    .agg(lambda x: x.mode().iloc[0])  # genre paling sering muncul
    .reset_index()
    .rename(columns={'genre': 'consistent_genre'})
)

# Gabungkan ke dataframe utama untuk menyamakan genre
df_anime.drop(columns='genre', inplace=True)
df_anime = df_anime.merge(genre_consensus, on='name')
df_anime.rename(columns={'consistent_genre': 'genre'}, inplace=True)


# mengecek apakah ada nama anime yang memiliki lebih dari 1 genre unik
genre_variation = (
    df_anime.groupby('name')['genre']
    .nunique()
    .reset_index()
    .rename(columns={'genre': 'unique_genre_count'})
)

# Filter nama yang punya lebih dari 1 genre berbeda
duplicate_genre_names = genre_variation[genre_variation['unique_genre_count'] > 1]

# Menampilkan hasil
print(f"Ada {len(duplicate_genre_names)} nama anime yang memiliki genre berbeda.")
print(duplicate_genre_names.head())
```

**output**:

![image](https://github.com/user-attachments/assets/8a952eef-2ae4-4c51-9a76-0eb73c866e8b)

Setelah proses mengambil genre pertama dari setiap name, dan menyamakan genre berdasarkan nama anime, dan memberikan output `0` atau `tidak ada name anime yang sama memiliki genre yang berbeda`. Tujuannya agar genre menjadi konsisten dan menghindari duplikasi atau inkonsistensi pada fitur genre.


6. **Menangani Rating -1**: Mengubah nilai rating -1 menjadi 0 untuk menandai anime yang ditonton tapi tidak diberi rating, sehingga memudahkan pemodelan tanpa kehilangan informasi interaksi pengguna.
```python
# 6. Menangani Rating -1 dengan mengganti rating -1 menjadi 0 (menandakan user menonton tanpa memberi rating)
df_rating['rating'] = df_rating['rating'].apply(lambda x: 0 if x == -1 else x)
```

7. **Mengecek Ulang Data**: Melakukan pengecekan awal pada data anime setelah preprocessing untuk memastikan transformasi berjalan sesuai harapan.
```python
# 7.Mengecek Ulang Data dataset anime_csv
df_anime.head()
```
**output**:

![image](https://github.com/user-attachments/assets/a3d78073-4491-42b5-8d82-096157001c71)

Kolom `name` pada df_anime kini bersih dari karakter khusus, menghasilkan data nama yang lebih konsisten dan berkualitas.

### Data Preprocessing & Preparation untuk Content-Based Filtering
**Berdasarkan SubBab `6.1.1 TF-IDF Vectorizer` pada file `ML_Terapan_Proyek2.ipynb`.**
#### 1. Persiapan Data untuk Content-Based Filtering
```python
# Menampilkan 5 data teratas pada kolom name dan genre
df_anime[['name', 'genre']].head()
```
**output**:

![image](https://github.com/user-attachments/assets/881eaddd-1703-4d43-bd18-cfe2705d35be)

Langkah awal Content-Based Filtering dimulai dengan menyiapkan data yang relevan, yaitu hanya kolom name dan genre dari dataset anime. Ini dilakukan karena sistem akan merekomendasikan anime berdasarkan kemiripan genre antar judul. Menampilkan 5 data teratas berguna untuk memastikan struktur data sudah sesuai sebelum diproses lebih lanjut.

#### 2. TF-IDF Vectorizer
```python
# Inisialisasi TF-IDF Vectorizer dengan tokenizer berbasis koma
tf_idf = TfidfVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',')])

# Melakukan fit pada data genre
tf_idf.fit(df_anime['genre'])

# Menampilkan nama-nama fitur (genre unik)
tf_idf.get_feature_names_out()
```
**output**:

![image](https://github.com/user-attachments/assets/567a5f3c-47aa-498f-bef0-519f7fe98583)


selanjutnya, menggunakan TF-IDF Vectorizer untuk mengubah data genre menjadi representasi numerik berbasis bobot kata. Karena genre dipisahkan oleh koma, digunakan tokenizer khusus untuk memecahnya. Setelah proses fit, model menghasilkan daftar genre unik sebagai fitur. Hasil ini akan digunakan untuk mengukur kemiripan antar anime berdasarkan genre-nya.

#### 3. Transformasi genre ke bentuk matriks TF-IDF
```python
# Transformasi genre ke bentuk matriks TF-IDF
tfidf_matrix = tf_idf.fit_transform(df_anime['genre'])

# Melihat ukuran matriks (baris = anime, kolom = genre unik)
print("Ukuran TF-IDF Matrix:", tfidf_matrix.shape)
```
**output**:

![image](https://github.com/user-attachments/assets/c6bc9782-ed72-4691-85a7-7adab8c74ea6)


Pada tahap ini, seluruh kolom genre dari dataset anime diubah menjadi matriks numerik menggunakan TF-IDF. Hasilnya adalah matriks berukuran 12017 baris (jumlah anime) dan 40 kolom (jumlah genre unik). Matriks ini merepresentasikan seberapa penting sebuah genre bagi setiap anime, dan menjadi dasar untuk menghitung kemiripan antar anime dalam sistem rekomendasi.


#### 4. Konversi ke bentuk matriks dense untuk keperluan visualisasi
```python
# Konversi ke bentuk matriks dense untuk keperluan visualisasi
dense_matrix = tfidf_matrix.todense()
print(dense_matrix)
```
**output**:

![image](https://github.com/user-attachments/assets/4c5f38ca-6aee-4c7e-9751-9b5a4b8992ec)


TF-IDF matrix yang awalnya berbentuk sparse dikonversi ke bentuk dense agar lebih mudah divisualisasikan dan dipahami. Setiap baris mewakili satu anime, dan setiap kolom menunjukkan bobot TF-IDF dari sebuah genre. Nilai 0 berarti genre tersebut tidak relevan bagi anime tersebut, sementara nilai 1.0 menunjukkan tingkat relevansi genre terhadap anime.

#### 5. menampilkan sebagian matriks TF-IDF sebagai DataFrame
```python
# menampilkan sebagian matriks TF-IDF sebagai DataFrame
tfidf_df = pd.DataFrame(
    dense_matrix,
    columns=tf_idf.get_feature_names_out(),
    index=df_anime['name']
)

# Menampilkan contoh 10 anime (baris) dan 22 genre acak (kolom)
tfidf_df.sample(10, axis=0).sample(22, axis=1)
```
**output**:

![image](https://github.com/user-attachments/assets/4b2adaea-06f0-4ea6-9cca-5889f45d3dde)


Pada tahap ini, mengonversi matriks TF-IDF ke dalam bentuk DataFrame agar lebih mudah dibaca dan dianalisis. Setiap baris mewakili anime berdasarkan nama, dan setiap kolom menunjukkan bobot genre tertentu. Kemudian, kita menampilkan sampel acak dari 10 anime dan 22 genre untuk melihat bagaimana genre diwakili dalam bentuk angka—nilai 1 berarti genre tersebut dominan pada anime itu, sementara 0 berarti tidak relevan. Ini akan menjadi dasar untuk menghitung kemiripan antar anime.

### Data Preprocessing & PreparationCollaborative Filtering
#### 1. Filtering ratings
```python
# Filtering ratings

# Hanya mengambil anime yang ada di df_anime (join berdasarkan anime_id)
df_rating = df_rating[df_rating['anime_id'].isin(df_anime['anime_id'])]

# Filter user dengan minimal 5 rating (mengurangi noise dan user yang sangat jarang aktif)
user_counts = df_rating['user_id'].value_counts()
active_users = user_counts[user_counts >= 5].index
df_rating = df_rating[df_rating['user_id'].isin(active_users)]

# Reset index setelah filter
df_rating.reset_index(drop=True, inplace=True)

# Info data setelah preprocessing
print("Shape df_anime:", df_anime.shape)
print("Shape df_rating:", df_rating.shape)
print("Sample df_rating:")
print(df_rating.head())
```
**output**:

![image](https://github.com/user-attachments/assets/a110b181-5935-4838-92ac-1f821280b785)

Di tahap Data Preparation Collaborative Filtering ini, pertama-tama data rating difilter agar hanya berisi anime yang memang ada di dataset anime (berdasarkan anime_id). Kemudian, hanya user yang memberi rating minimal 5 kali yang diikutsertakan, supaya data lebih bersih dan mengurangi noise dari user yang jarang aktif. Setelah itu, index data di-reset agar rapi. Dengan langkah ini, dataset rating siap untuk digunakan dalam pemodelan Collaborative Filtering.

#### 2. Label Encoding
```python
# Label Encoding
# Mengubah user_id dan anime_id menjadi representasi numerik berurutan (0, 1, 2, ...)
user_encoder = LabelEncoder()
anime_encoder = LabelEncoder()

df_rating['user'] = user_encoder.fit_transform(df_rating['user_id'])
df_rating['anime'] = anime_encoder.fit_transform(df_rating['anime_id'])

# Menyimpan jumlah unik user dan anime untuk input dimensi pada embedding layer model
num_users = df_rating['user'].nunique()
num_anime = df_rating['anime'].nunique()

print(f"Jumlah user unik: {num_users}")
print(f"Jumlah anime unik: {num_anime}")
print("\nDataFrame df_rating setelah label encoding:")
print(df_rating.head())
```
**output**:

![image](https://github.com/user-attachments/assets/866182e5-3503-438a-a6b2-a84f4229531f)

Selanjutnya Langkah Label Encoding ini mengubah user_id dan anime_id yang awalnya berupa angka acak atau ID asli menjadi representasi numerik berurutan mulai dari 0. Ini penting agar data bisa langsung dipakai sebagai input untuk embedding layer di model Collaborative Filtering. Setelah proses ini, kita tahu ada 1446 user unik dan 3296 anime unik yang siap diproses di model. Data df_rating kini memiliki kolom tambahan user dan anime yang berisi ID numerik baru untuk keperluan pemodelan.

#### 3. Membagi Data untuk Training dan Validasi
```python
# Membagi Data untuk Training dan Validasi
df_shuffled = df_rating.sample(frac=1, random_state=42).reset_index(drop=True)
print("\nDataFrame df_rating setelah diacak:")
print(df_shuffled.head())

#  min dan max rating untuk normalisasi
min_rating = df_shuffled['rating'].min()
max_rating = df_shuffled['rating'].max()

# Membuat variabel x untuk mencocokkan data user dan anime menjadi satu value
x = df_shuffled[['user', 'anime']].values

# Membuat variabel y untuk membuat rating dari hasil
# Normalisasi rating ke rentang 0-1
y = df_shuffled['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * df_shuffled.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(f"\nBentuk x_train: {x_train.shape}")
print(f"Bentuk y_train: {y_train.shape}")
print(f"Bentuk x_val: {x_val.shape}")
print(f"Bentuk y_val: {y_val.shape}")

# Menampilkan sebagian kecil dari data x dan y (untuk verifikasi)
print("\nContoh x (user, anime):")
print(x[:5])
print("\nContoh y (normalized rating):")
print(y[:5])
```
**output**:

![image](https://github.com/user-attachments/assets/9e5cf097-8fc8-4451-960d-cd91e4e9b79d)

Data rating diacak lalu rating dinormalisasi ke rentang 0-1. Data user dan anime digabungkan sebagai fitur input (x), dan rating sebagai target (y). Data dibagi 80% untuk training dan 20% untuk validasi, menghasilkan masing-masing sekitar 7.473 dan 1.869 data, siap untuk pelatihan model collaborative filtering.


## Model Solution & Result
***Notes**: karena kode Modeling terlalu panjang dan banyak, maka tidak memungkinkan untuk ditampilkan pada laporan. Kode lengkap berada pada file `ML_Terapan_Proyek2.ipynb`.*

### 1. Content-Based Filtering
Model Content-Based Filtering memiliki kelebihan dalam memberikan rekomendasi yang personal dan relevan karena didasarkan pada karakteristik item yang disukai pengguna, serta tidak memerlukan data dari pengguna lain sehingga cocok untuk sistem dengan jumlah pengguna yang masih terbatas. Namun, model ini juga memiliki kekurangan, seperti terbatasnya cakupan rekomendasi karena hanya menyarankan item yang mirip dengan yang sudah diketahui, serta kesulitan menangani cold start untuk item yang minim informasi.


#### Result Mendapatkan Rekomendasi dengan Content-Based Filtering
Pada tahap ini, dibuat fungsi anime_recommendations yang berfungsi untuk memberikan rekomendasi anime yang mirip berdasarkan nama anime yang diberikan sebagai input. Fungsi ini memanfaatkan matriks cosine similarity yang telah dihitung sebelumnya untuk mencari k anime dengan nilai kemiripan tertinggi terhadap anime yang dicari. Nilai k diatur sebesar 5, artinya sistem akan menampilkan 5 rekomendasi teratas. Fungsi ini juga memastikan bahwa anime yang dijadikan acuan tidak termasuk dalam daftar rekomendasi, sehingga hanya anime lain yang relevan secara genre yang ditampilkan.

##### Menyajikan top-N recommendation sebagai output.
![image](https://github.com/user-attachments/assets/07652e04-0873-4168-b969-e2c32ecfe817)

Sebagai contoh, sistem digunakan untuk mencari anime yang mirip dengan "Kimi no Na wa", yang memiliki genre Drama. Sistem mengecek baris dari anime tersebut dalam dataset, lalu menghitung kemiripan genre dengan seluruh anime lainnya menggunakan nilai cosine similarity dari data TF-IDF. Hasilnya adalah 5 rekomendasi anime yang memiliki genre yang mirip,dalam hal ini drama-dengan "Kimi no Na wa", memberikan saran tontonan yang relevan berdasarkan konten.



### 2. Collaborative Filtering
Model Collaborative Filtering memiliki kelebihan utama dalam kemampuannya memberikan rekomendasi yang bersifat personalized dengan memanfaatkan pola interaksi antar pengguna dan item, tanpa perlu mengetahui atribut detail dari item (seperti genre atau deskripsi). Model ini juga mampu menemukan hubungan tersembunyi antar item berdasarkan preferensi pengguna. Namun, kekurangannya terletak pada masalah cold start, di mana model sulit memberikan rekomendasi untuk pengguna baru atau item baru yang belum memiliki cukup interaksi, serta tergantung pada ketersediaan data rating yang cukup besar dan beragam.


#### Result Mendapatkan Rekomendasi dengan Collaborative Filtering
Pada tahap Result Collaborative Filtering, sistem rekomendasi mulai dengan menyalin df_anime menjadi anime_df sebagai basis data utama untuk penggabungan dan penampilan hasil. Selanjutnya, sistem mengambil secara acak satu user_id dari data rating dan menentukan riwayat tontonan user tersebut dengan memfilter semua anime yang sudah pernah dinilai. Setelah itu, sistem mengidentifikasi anime yang belum pernah ditonton oleh user dengan membandingkan anime_id yang ada di dataset dengan riwayat user, lalu mengubahnya ke bentuk encoded menggunakan anime_encoder. Jika tidak ada kandidat, sistem memilih 10 anime acak sebagai alternatif. Kemudian, sistem membentuk array pasangan user-anime encoded sebagai input model, dan menggunakan model rekomendasi untuk memprediksi rating dari setiap anime yang belum ditonton oleh user. Hasil prediksi di-flatten menjadi array satu dimensi agar mudah diproses. Terakhir, sistem memilih 10 anime dengan prediksi rating tertinggi untuk direkomendasikan, mengubah kembali ID encoded ke ID asli, dan menampilkan 5 anime terbaik yang sudah ditonton user serta 10 rekomendasi anime teratas yang belum ditonton lengkap dengan nama, genre, dan nilai prediksi rating-nya.

##### Menyajikan top-N(10) recommendation sebagai output.
![image](https://github.com/user-attachments/assets/86c6061c-8fc3-4697-afed-1b2bd2ec498d)

Hasil di atas adalah rekomendasi untuk user dengan ID 46801. Dari output tersebut, dapat dilihat bahwa anime yang telah diberi rating tinggi oleh user didominasi oleh genre Action dan Comedy, dengan tambahan genre lain seperti Mystery dan Ecchi. Contohnya, Mirai Nikki Redial (Action), Kono Naka ni Hitori Imouto ga Iru (Comedy), dan No Game No Life Specials (Ecchi) menunjukkan bahwa user memiliki ketertarikan terhadap anime yang penuh aksi, humor, dan elemen hiburan ringan.

Sementara itu, daftar Top 10 rekomendasi anime untuk user ini sebagian besar mengandung genre Action, dengan tambahan genre seperti Adventure, Comedy, dan Drama. Beberapa anime yang direkomendasikan seperti Hunter x Hunter 2011, Gintama039, dan Kyoukaisenjou no Horizon menunjukkan bahwa model berupaya menyarankan anime dengan tema dan nuansa yang mirip dengan preferensi user. Dengan kata lain, model collaborative filtering berhasil menyesuaikan rekomendasi berdasarkan pola kesukaan user sebelumnya, sehingga hasil rekomendasi bersifat relevan dan berpotensi disukai.


## Evaluation
### Evaluasi Content-Based
**Metrik Evaluasi Sistem Rekomendasi: Recommender System precision**

Presisi adalah metrik yang digunakan untuk mengukur seberapa relevan item yang direkomendasikan oleh sistem. Ini dihitung sebagai rasio jumlah rekomendasi yang relevan dengan total jumlah item yang direkomendasikan.

Rumus untuk presisi (`P`) adalah sebagai berikut:

$$
P = \frac{\text{of our recommendations that are relevant}}{\text{of items we recommended}}
$$

Di mana:
- $\text{of our recommendations that are relevant}$: Jumlah item yang direkomendasikan oleh sistem yang benar-benar relevan bagi pengguna.
- $\text{of items we recommended}$: Total jumlah item yang direkomendasikan oleh sistem kepada pengguna.

**Contoh: Berdasarkan Hasil SubBab `6.1.3 Mendapatkan Rekomendasi` pada file `ML_Terapan_Proyek2.ipynb`.**
![image](https://github.com/user-attachments/assets/07652e04-0873-4168-b969-e2c32ecfe817)

Jika sistem rekomendasi merekomendasikan 5 film, dan ke5 film tersebut relevan bagi pengguna, maka presisinya adalah:

$$
P = \frac{5}{5} = 1
$$

Ini berarti **100%** dari rekomendasi adalah relevan.

### Evaluasi Collaborative Filtering
**Metrik Evaluasi Collaborative Filtering: MSE & RMSE**

Dalam sistem rekomendasi berbasis Collaborative Filtering, Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE) adalah metrik umum yang digunakan untuk mengevaluasi akurasi prediksi model. Keduanya mengukur rata-rata kuadrat atau akar kuadrat dari kesalahan (selisih antara nilai prediksi dan nilai sebenarnya).

---

#### Mean Squared Error (MSE)

MSE mengukur rata-rata dari kuadrat error (selisih) antara nilai yang diprediksi dan nilai aktual. Karena mengkuadratkan error, MSE memberikan bobot yang lebih besar pada error yang besar, sehingga sensitif terhadap outlier.

**Rumus MSE:**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Di mana:
* $n$: Jumlah observasi (data).
* $y_i$: Nilai aktual (misalnya, rating sebenarnya yang diberikan pengguna).
* $\hat{y}_i$: Nilai prediksi (misalnya, rating yang diprediksi oleh model).

---

#### Root Mean Squared Error (RMSE)

RMSE adalah akar kuadrat dari MSE. Ini lebih mudah diinterpretasikan karena unitnya sama dengan unit variabel output (misalnya, unit rating). RMSE juga sensitif terhadap error besar, sama seperti MSE.

**Rumus RMSE:**

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} = \sqrt{\text{MSE}}
$$

Di mana:
* $n$: Jumlah observasi (data).
* $y_i$: Nilai aktual.
* $\hat{y}_i$: Nilai prediksi.

#### Contoh Implementasi dan Hasil Evaluasi

Dalam konteks model Collaborative Filtering , menghitung MSE dan RMSE setelah proses pelatihan. Berikut adalah contoh kode dan output yang menunjukkan hasil evaluasi:

```python
# 'x_val' dan 'y_val' adalah data validasi dan labelnya
results = model.evaluate(x_val, y_val, verbose=1)

print("\nHasil Evaluasi pada Data Validasi:")
print(f"Loss (MSE): {results[0]:.4f}")
print(f"RMSE: {results[1]:.4f}")
```
#### Hasil Evaluasi dengan MSE & RMSE
![image](https://github.com/user-attachments/assets/9bc6268c-1b9f-4f43-8bff-9f54f1a62624)

Hasil evaluasi Collaborative Filtering menunjukkan performa model pada data validasi dengan nilai loss (MSE) sebesar sekitar 0.0599 dan RMSE sekitar 0.2445. Nilai MSE yang kecil menunjukkan bahwa rata-rata kuadrat selisih antara rating asli dan prediksi model cukup rendah, artinya model mampu memprediksi rating dengan akurasi yang baik.

#### Visualisasi Ecpoch Tarining & Validation
![image](https://github.com/user-attachments/assets/70e70fac-64d9-4950-ad59-1f0a01302395)

Visualisasi pada gambar memperlihatkan proses pelatihan model selama 100 epoch. Terlihat bahwa nilai RMSE baik pada data training maupun validation mengalami penurunan yang konsisten dari epoch pertama hingga epoch ke-100, menandakan bahwa model mengalami peningkatan performa secara bertahap. Pada akhir pelatihan, di epoch ke-100, nilai RMSE mencapai sekitar 0.1 untuk training dan sekitar 0.2 untuk validation, yang menunjukkan bahwa model berhasil melakukan generalisasi dengan baik tanpa overfitting yang signifikan.

## Penutup & Kesimpulan
Proyek ini berhasil mengembangkan sistem rekomendasi anime dengan mengimplementasikan dua algoritma utama, yaitu Content-Based Filtering (CBF) dan Collaborative Filtering (CF). Model Content-Based Filtering menunjukkan performa yang sangat baik dengan akurasi mencapai 100%, sedangkan model Collaborative Filtering berhasil memberikan evaluasi yang memuaskan dengan nilai MSE sebesar 0.599 dan RMSE sebesar 0.2445. Kedua model tersebut mampu menghasilkan rekomendasi yang relevan, dengan CBF menampilkan top-5 rekomendasi dan CF memberikan top-10 rekomendasi bagi pengguna.

Keberhasilan ini membuktikan bahwa masing-masing pendekatan dapat memberikan kontribusi signifikan dalam sistem rekomendasi anime. Namun, untuk meningkatkan kualitas dan akurasi rekomendasi secara keseluruhan, pengembangan lebih lanjut disarankan dengan menggabungkan kedua metode tersebut dalam sebuah sistem Hybrid Filtering. Pendekatan hybrid ini diharapkan dapat memanfaatkan kelebihan masing-masing metode sekaligus mengatasi keterbatasan, seperti masalah cold start dan sparsity data, sehingga menghasilkan rekomendasi yang lebih personal dan tepat sasaran bagi pengguna.

---
Salam,

Maxwell Massie

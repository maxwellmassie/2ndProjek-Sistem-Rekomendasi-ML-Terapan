# Laporan Proyek Machine Learning - Gold Stein Maxwell Massie
## Sistem Rekomendasi Anime

## Project Overview

### Latar Belakang
Pertumbuhan pesat industri hiburan digital, khususnya anime, menyebabkan kebutuhan akan sistem rekomendasi yang mampu membantu pengguna menemukan konten sesuai preferensi mereka. Dengan ribuan judul anime dan beragam selera pengguna, sistem rekomendasi menjadi solusi penting untuk meningkatkan pengalaman pengguna.

Metode Content-Based Filtering yang memanfaatkan fitur konten seperti genre dan Collaborative Filtering yang menggunakan interaksi pengguna terbukti efektif dalam merekomendasikan item yang relevan. Penggabungan kedua metode ini membantu mengatasi keterbatasan masing-masing dan memberikan rekomendasi yang lebih akurat.

Menurut Su dan Khoshgoftaar (2019), Collaborative Filtering adalah teknik yang paling banyak digunakan dalam sistem rekomendasi karena kemampuannya menangkap pola preferensi pengguna dari data interaksi [1]. Di Indonesia, penelitian oleh Faurina dan Sitanggang (2023) menerapkan Content-Based Filtering dan Collaborative Filtering pada sistem rekomendasi destinasi wisata di Bali. Penelitian ini menunjukkan bahwa pendekatan hybrid tersebut efektif dalam memberikan rekomendasi personal dengan performa evaluasi yang baik [2].

Proyek ini bertujuan membangun sistem rekomendasi anime menggunakan dataset MyAnimeList, memanfaatkan teknik Content-Based Filtering dan Collaborative Filtering untuk memberikan rekomendasi yang personal dan relevan.

**Referensi**

[1] X. Su dan T. M. Khoshgoftaar, “A survey of collaborative filtering techniques,” Advances in Artificial Intelligence, vol. 2019, Article ID 4214258, 2019.
Tersedia online: https://doi.org/10.1155/2019/4214258

[2] R. Faurina dan E. Sitanggang, “Implementasi Metode Content-Based Filtering dan Collaborative Filtering pada Sistem Rekomendasi Wisata di Bali,” Techno.COM, vol. 22, no. 4, pp. 870-881, Nov. 2023.
Tersedia online: https://core.ac.uk/download/pdf/595469606.pdf


## Business Understanding

Dalam era digital saat ini, pengguna anime menghadapi tantangan dalam memilih judul yang sesuai dari ribuan pilihan yang tersedia. Sistem rekomendasi yang efektif sangat dibutuhkan untuk membantu pengguna menemukan anime yang sesuai dengan preferensi mereka berdasarkan interaksi dan karakteristik konten.

### Problem Statements
1. Bagaimana membantu pengguna menemukan anime yang sesuai dengan preferensi mereka dari data rating dan fitur konten yang tersedia?
2. Bagaimana membangun model rekomendasi yang akurat dan efisien dengan memanfaatkan data interaksi pengguna dan informasi anime?

### Goals
1. Mengembangkan sistem rekomendasi anime yang memberikan rekomendasi personal berbasis preferensi pengguna dan karakteristik konten anime.
2. Memanfaatkan data rating pengguna dan fitur konten anime untuk menghasilkan rekomendasi yang akurat dan relevan. Dengan memproses data interaksi dan deskripsi konten, sistem akan memberikan rekomendasi berdasarkan kesamaan konten dan pola rating.
3. Membangun model rekomendasi dengan metode Content-Based Filtering dan Collaborative Filtering dengan Matrix Factorization untuk meningkatkan akurasi rekomendasi.

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
* **anime_id** (`int64`): ID unik untuk setiap anime dari situs MyAnimeList. Digunakan sebagai penghubung dengan data rating.
* **name** (`object`): Nama lengkap atau judul dari anime.
* **genre** (`object`): Daftar genre dari anime, dipisahkan dengan koma (misalnya: Action, Adventure, Drama, Fantasy, Magic,).
* **type** (`object`): Tipe tayangan dari anime (misalnya TV, Movie, OVA, dll).
* **episodes** (`object`): Jumlah episode dari anime. Disimpan sebagai teks karena beberapa nilai bisa berupa 'Unknown' atau tidak terisi secara numerik.
* **rating** (`float64`): Rata-rata rating dari pengguna terhadap anime tersebut dalam skala 1–10.
* **members** (`int64`): Jumlah anggota komunitas yang memasukkan anime ke dalam daftar mereka. Mewakili tingkat popularitas anime.

#### 📁 `Rating.csv` (7.813.737 data interaksi pengguna)
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



## Data Preparation
# **5. Data Preprocessing & Preparation**
1. **Sampling Data Rating**: Karena dataset rating sangat besar, maka mengambil sampel 50.000 baris secara acak untuk mempercepat proses dan mengurangi beban komputasi, sambil tetap menjaga representasi data.
```python
# 1. Sampling Data Rating
df_rating = df_rating.sample(n=50000, random_state=42).reset_index(drop=True)

# mengecek info setelah sampling
print(df_rating.info())

# mengecek beberapa baris awal
print(df_rating.head())
```

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

4. **Menghapus Data Duplikat**: Menghilangkan baris duplikat pada data rating untuk menjaga integritas data dan menghindari bias berlebih dalam model rekomendasi.
```python
# 4. Menghapus Data Duplikat
df_rating.drop_duplicates(inplace=True)

print("\nDuplicate Rows in df_rating:")
print(df_rating.duplicated().sum())
```

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

## Model Solution & Result
*Notes: karena kode Modeling terlalu panjang dan banyak, maka tidak memungkinkan untuk ditampilkan pada laporan. Kode lengkap berada pada file `ML_Terapan_Proyek2.ipynb`.*

### 1. Content-Based Filtering
Model Content-Based Filtering memiliki kelebihan dalam memberikan rekomendasi yang personal dan relevan karena didasarkan pada karakteristik item yang disukai pengguna, serta tidak memerlukan data dari pengguna lain sehingga cocok untuk sistem dengan jumlah pengguna yang masih terbatas. Namun, model ini juga memiliki kekurangan, seperti terbatasnya cakupan rekomendasi karena hanya menyarankan item yang mirip dengan yang sudah diketahui, serta kesulitan menangani cold start untuk item yang minim informasi.

Dalam implementasinya, model ini dibangun untuk merekomendasikan anime berdasarkan kemiripan genre. Proses dimulai dengan menyiapkan data yang hanya mencakup kolom name dan genre, karena genre menjadi dasar perhitungan kemiripan. Genre diolah menggunakan TF-IDF Vectorizer dengan tokenizer khusus untuk memecah genre yang dipisahkan koma, menghasilkan representasi numerik dari setiap genre. Hasilnya adalah matriks TF-IDF berukuran 12017×40 yang menunjukkan bobot pentingnya setiap genre bagi tiap anime. Matriks ini kemudian dikonversi ke dalam DataFrame agar lebih mudah dianalisis, di mana setiap nilai menggambarkan kekuatan representasi sebuah genre pada anime tertentu. Selanjutnya, digunakan cosine similarity untuk mengukur tingkat kemiripan antar anime berdasarkan vektor genre-nya. Hasilnya adalah matriks similarity 12017×12017 yang menunjukkan seberapa mirip dua anime dalam hal genre, dan menjadi dasar dalam menghasilkan rekomendasi anime yang relevan secara konten.

#### Result Mendapatkan Rekomendasi dengan Content-Based Filtering
Pada tahap ini, dibuat fungsi anime_recommendations yang berfungsi untuk memberikan rekomendasi anime yang mirip berdasarkan nama anime yang diberikan sebagai input. Fungsi ini memanfaatkan matriks cosine similarity yang telah dihitung sebelumnya untuk mencari k anime dengan nilai kemiripan tertinggi terhadap anime yang dicari. Nilai k diatur sebesar 5, artinya sistem akan menampilkan 5 rekomendasi teratas. Fungsi ini juga memastikan bahwa anime yang dijadikan acuan tidak termasuk dalam daftar rekomendasi, sehingga hanya anime lain yang relevan secara genre yang ditampilkan.

##### Menyajikan top-N recommendation sebagai output.
![image](https://github.com/user-attachments/assets/07652e04-0873-4168-b969-e2c32ecfe817)

Sebagai contoh, sistem digunakan untuk mencari anime yang mirip dengan "Kimi no Na wa", yang memiliki genre Drama. Sistem mengecek baris dari anime tersebut dalam dataset, lalu menghitung kemiripan genre dengan seluruh anime lainnya menggunakan nilai cosine similarity dari data TF-IDF. Hasilnya adalah 5 rekomendasi anime yang memiliki genre yang mirip—dalam hal ini drama—dengan "Kimi no Na wa", memberikan saran tontonan yang relevan berdasarkan konten.



### 2. Collaborative Filtering

## Evaluation
### Evaluasi Content-Based
**Metrik Evaluasi Sistem Rekomendasi: Presisi**

Presisi adalah metrik yang digunakan untuk mengukur seberapa relevan item yang direkomendasikan oleh sistem. Ini dihitung sebagai rasio jumlah rekomendasi yang relevan dengan total jumlah item yang direkomendasikan.

Rumus untuk presisi (`P`) adalah sebagai berikut:

$$
P = \frac{\text{of our recommendations that are relevant}}{\text{of items we recommended}}
$$

Di mana:
- $\text{of our recommendations that are relevant}$: Jumlah item yang direkomendasikan oleh sistem yang benar-benar relevan bagi pengguna.
- $\text{of items we recommended}$: Total jumlah item yang direkomendasikan oleh sistem kepada pengguna.

**Contoh: Berdasarkan Hasil SubBab `6.1.3 Mendapatkan Rekomendasi`**
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
# Asumsi 'model' adalah model Collaborative Filtering Anda
# Asumsi 'x_val' dan 'y_val' adalah data validasi dan labelnya
results = model.evaluate(x_val, y_val, verbose=1)

print("\nHasil Evaluasi pada Data Validasi:")
print(f"Loss (MSE): {results[0]:.4f}")
print(f"RMSE: {results[1]:.4f}")
```


**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.


---
Salam,

Maxwell Massie

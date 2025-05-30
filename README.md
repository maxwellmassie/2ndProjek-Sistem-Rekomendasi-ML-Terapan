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
#### Menampilkan Statistik Deskriptif Rating.csv

#### Mengecek & Menampilkan Missing Value 
#### Mengecek & Menampilkan Data Duplicate

#### Visualisasi Top 10 Genre Anime
#### Visualisasi Berdasarkan type Anime


## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

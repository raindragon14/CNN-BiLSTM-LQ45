# Catatan Keputusan Desain

> **Peran:** satu-satunya sumber jejak keputusan -> dasar (artikel + lokasi) dan
> lokasi implementasi.
> **Audiens:** publik.
> **Bukan untuk:** analisis mendalam atau alternatif yang ditolak (lihat
> `internal/First_Principles_Parameter_Analysis.md`) atau naskah skripsi
> (`internal/SKRIPSI.md`).

Berkas ini adalah jejak setiap keputusan penelitian: apa yang diputuskan, atas
dasar apa, dan di mana diterapkan. Tujuannya agar keputusan dapat ditelusuri
kembali tanpa mengandalkan ingatan.

## Cara membaca

- **Dasar** menyebut artikel dan lokasi di dalam artikel (bagian, tabel, atau
  persamaan), bukan hanya nama penulis.
- **Implementasi** menyebut berkas dan kunci yang memuat keputusan tersebut.
- Status: `final` (sudah diputuskan) atau `menunggu` (belum diputuskan).

---

## Data

| Keputusan | Nilai | Dasar | Implementasi | Status |
|---|---|---|---|---|
| Periode | 2018-01-01 s.d. 2025-12-31 | 8 tahun; mencakup COVID 2020 dan kenaikan suku bunga 2022-2024 | `configs/data.yaml` `period` | final |
| Pool kandidat | 45 konstituen LQ45 periode Agu 2019 - Jan 2020 | IDX LQ45 Company Profiles (Agu 2019), halaman *Contents*; arsip Wayback | `configs/universe.yaml` | final |
| Alasan pool dibekukan | Menghindari pemilihan arbitrer dan bias survivorship dari daftar akhir periode | - | `data/README.md` | final |
| Sumber harga | Yahoo Finance `.JK`, `adjusted close` | Sumber sekunder; keterbatasan diungkap | `configs/data.yaml` `sources.price` | final |
| Sumber kurs | JISDOR, Bank Indonesia | Kurs referensi resmi Bank Indonesia | `configs/data.yaml` `sources.fx` | final |
| Sumber suku bunga | BI-7DRRR, Bank Indonesia | Halaman indikator resmi BI | `configs/data.yaml` `sources.macro` | final |
| Tolok ukur | IHSG (`.JKSE`) | Pembanding beli-dan-tahan | `configs/data.yaml` `sources.benchmark` | final |
| Biaya transaksi | beli 0,19%, jual 0,29%, lot 100 | Ketentuan sekuritas ritel Indonesia | `configs/data.yaml` `costs` | final |
| Penanganan data kosong | forward-fill maks. 3 hari | Kalender bursa tidak sinkron | `configs/data.yaml` `cleaning` | final |
| Bentuk split | Train awal 2 tahun, jendela *expanding* | Walk-forward standar; jendela melebar mempertahankan seluruh data historis. Prinsip, bukan sitasi | `configs/split.yaml` | final |
| Purge & embargo | purge 5 hari (=tau), embargo 59 hari (=w-1) | Lopez de Prado (2018) Bab 7 (purged cross-validation): buang label latih yang tumpang tindih dengan validation, dan beri jeda setelah test | `configs/split.yaml` | final |
| Panjang validation/test/step | 63 / 21 / 21 hari | Keputusan mandiri: ~1 kuartal, 1 bulan, 1 bulan; selaras hold bulanan dan tau=5 | `configs/split.yaml` | final |
| Penyelarasan makro | Tanggal publikasi | Prinsip: mencegah kebocoran informasi ke depan | `configs/data.yaml` `macro_lag` | final |
| Jenis return | log-return | Sen & Dutta (2021) menghitung log return; sifat aditif antar waktu | `configs/data.yaml` `cleaning.return_type` | final |

## Fitur

| Keputusan | Nilai | Dasar | Implementasi | Status |
|---|---|---|---|---|
| Representasi harga | Level `Adj Close` (bukan log-return) | Chaweewanchon & Chaysiri (2022) Bagian 4.1.1 (harga penutupan); Sebastian & Tantia (2024) Bagian data (6 fitur level) | `configs/experiment.yaml` | final |
| Fitur dasar | `close`, `volume` | Sebastian & Tantia (2024) Bagian data (OHLC + adjusted close + volume); Sen & Dutta (2021) Bagian data (6 fitur) | `configs/experiment.yaml` | final |
| RSI | periode 14 | Espiga-Fernandez et al. (2024) Lampiran B.3 | `configs/experiment.yaml` | final |
| CCI | periode 20 | Espiga-Fernandez et al. (2024) Lampiran B.4 | `configs/experiment.yaml` | final |
| CMO | periode 14 | Espiga-Fernandez et al. (2024) Lampiran B.5 | `configs/experiment.yaml` | final |
| MFI | periode 14 (butuh harga + volume) | Espiga-Fernandez et al. (2024) Lampiran B.6 | `configs/experiment.yaml` | final |
| Penyederhanaan | AROONOSC, Williams %R, STOCHF dari Espiga-Fernandez et al. tidak diambil | Dikurangi sadar agar jumlah fitur tetap 8 | `configs/experiment.yaml` | final |
| Makro: BI-7DRRR | level (%) | Ekstensi; selisih bulanan hampir selalu nol sehingga tidak informatif | `configs/experiment.yaml` | final |
| Makro: JISDOR | log-return harian | Tidak ada artikel yang memakai makro sebagai fitur; level JISDOR non-stasioner, sehingga dipakai log-return agar stasioner | `configs/experiment.yaml` | final |
| Jumlah fitur | 8 | Rentang literatur 6 fitur (Sebastian & Tantia 2024; Sen & Dutta 2021) sampai 11 fitur (Espiga-Fernandez et al. 2024) | `configs/experiment.yaml` | final |
| Penyesuaian harga | Kolom OHLC diskalakan dengan faktor `Adj Close / Close` | Mencegah lompatan akibat dividen dan pemecahan saham; Sen & Dutta (2021) memakai `adjusted_close` | `src/lq45/features/build.py` | final |
| Kalender dan data kosong | Kalender bursa dari IHSG; forward-fill maks. 3 hari | `configs/data.yaml` `cleaning` | `src/lq45/features/build.py` | final |

## Praproses

| Keputusan | Nilai | Dasar | Implementasi | Status |
|---|---|---|---|---|
| Transformasi robust | Winsorization ambang `[Q1 - 1,5*IQR, Q3 + 1,5*IQR]` | Sebastian & Tantia (2024) Bagian data pre-processing (outlier via boxplot, dicapit) | `configs/experiment.yaml` | final |
| Skala | Min-max scaler | Sebastian & Tantia (2024) Bagian data pre-processing | `configs/experiment.yaml` | final |
| Cakupan transformasi | Fitur **dan** target | Sebastian & Tantia (2024) memperlakukan outlier pada keseluruhan data sebelum pemodelan | `configs/experiment.yaml` | final |
| Titik *fit* | Hanya data latih tiap jendela walk-forward | **Prinsip**, bukan sitasi: pencegahan kebocoran informasi ke depan | `src/lq45/features/preprocess.py` | final |
| Alternatif yang ditolak | Huber location (Chaweewanchon & Chaysiri 2022 Bagian 3.5.2, `k=1,435`) | Artikel tidak menjelaskan standardisasi sebelum Huber, sehingga `k=1,435` pada harga mentah praktis menjadi median; sulit direplikasi persis | `docs/keputusan_desain.md` | final |

## Target dan Pelatihan

| Keputusan | Nilai | Dasar | Implementasi | Status |
|---|---|---|---|---|
| Target | Log-return 5 hari ke depan | Selaras horizon `tau=5` dan hold bulanan | `configs/model.yaml` | final |
| Fungsi loss | MSE | Chaweewanchon & Chaysiri (2022) Bagian pemilihan hyperparameter ("Mean Squared Error (MSE) was used as the loss function"); Kim et al. (2025) ("standard loss function (e.g., mean squared error)") | `configs/model.yaml` | final |
| Alternatif loss yang ditolak | Huber loss | Tidak dipakai artikel mana pun di folder; kebutuhan kokoh sudah ditangani winsorization (Sebastian & Tantia 2024); MV menuntut estimasi rata-rata, bukan median | `docs/keputusan_desain.md` | final |
| Jendela lihat-balik | `w=60` | Keputusan mandiri (praktik umum, kompromi bias-varians); diuji pada sensitivitas jendela | `configs/model.yaml` | final |
| Horizon prediksi | `tau=5` | Keputusan mandiri (selaras hold bulanan); diuji pada sensitivitas jendela | `configs/model.yaml` | final |
| Jendela estimasi kovarians | `L=120` | Tidak ada artikel yang menetapkannya; diuji sensitivitas pada tahap portofolio (L diuji pada 60, 120, 252) | `configs/portfolio.yaml` | final |
| Seed averaging | 0, 1, 2, 3, 4 | Redam variansi pelatihan (keputusan mandiri) | `configs/model.yaml` | final |
| Learning rate | 0,0001 | Chaweewanchon & Chaysiri (2022) Bagian 4.1.3; artikel itu juga mengutip Hastie et al. (2017) bahwa lr < 0,01 | `configs/model.yaml` | final |
| Optimizer | Adam | Chaweewanchon & Chaysiri (2022) Bagian 4.1.3; Sebastian & Tantia (2024) | `configs/model.yaml` | final |
| Jumlah epoch | 100 | Chaweewanchon & Chaysiri (2022) Bagian 4.1.3 (pelatihan berhenti pada 100-120 epoch); Sebastian & Tantia (2024) | `configs/model.yaml` | final |
| Early stopping | patience 8 | Sebastian & Tantia (2024) (berhenti setelah 8 epoch tanpa perbaikan) | `configs/model.yaml` | final |
| Dropout | 0,2 seragam (CNN, LSTM, Dense) | Sebastian & Tantia (2024) (dropout 0,2); angka 0,3 sebelumnya tidak berdasar | `configs/model.yaml` | final |
| Fungsi aktivasi | ReLU | Espiga-Fernandez et al. (2024) Tabel 3 (ReLU pada kedua lapis konvolusi dan lapis Linear) | `configs/model.yaml` | final |
| Filter CNN | Dua lapis 32 -> 64 | Espiga-Fernandez et al. (2024) Tabel 3 memakai pola persis ini | `configs/model.yaml` | final |
| Kernel CNN | 3 | Chaweewanchon & Chaysiri (2022) 3x3 | `configs/model.yaml` | final |
| Unit BiLSTM | 64 | Tidak ada artikel yang menetapkannya; disetel pada periode desain. Grid sengaja mencakup nilai Chaweewanchon & Chaysiri (2022) = 128 dan Sebastian & Tantia (2024) = 32 | `configs/model.yaml` | final |
| Lapis BiLSTM | 2 | Graves, Mohamed & Hinton (2013) memperkenalkan RNN berlapis; Chaweewanchon & Chaysiri (2022) dan Sebastian & Tantia (2024) memakai dua lapis | `configs/model.yaml` | final |
| Peluruhan bobot | 0 | Tidak ada artikel yang menetapkannya; disetel pada periode desain (0 dibanding 0,0001) | `configs/model.yaml` | final |
| Pooling | MaxPool 2 | Artikel menyebut pooling tanpa angka seragam; disetel pada periode desain (2 dibanding 3) | `configs/model.yaml` | final |
| BatchNorm | Ya | Chaweewanchon & Chaysiri (2022) memakai BatchNorm | `configs/model.yaml` | final |
| Perangkat | `auto` (CUDA bila tersedia, jika tidak CPU) | Jalur resmi CPU menjaga klaim hemat sumber daya dan reproducibility; GPU hanya eksperimen tambahan | `configs/model.yaml` | final |
| Batch size | 32 | Tidak ada artikel yang menetapkannya; disetel pada periode desain (32 dibanding 64). Huang et al. (2024) memakai jumlah saham sebagai batch | `configs/model.yaml` | final |
| Cakupan model | Satu model bersama semua saham | Espiga-Fernandez et al. (2024) masukan `lookback x instrumen x fitur`; Huang et al. (2024) masukan `32 x 15`; sekaligus membuat pelatihan layak di CPU | `configs/model.yaml` | final |
| Prosedur item lemah | Setel unit BiLSTM, peluruhan bobot, pooling, dan batch size pada periode desain (latih 2018, validasi 2019), lalu bekukan; laporkan grid 24 kombinasi sebagai lampiran | Mengubah asumsi menjadi prosedur pemilihan yang dapat diaudit; mencegah kebocoran karena tidak menyentuh data OOS. Grid unit sengaja mencakup nilai artikel (32 = Sebastian & Tantia 2024, 128 = Chaweewanchon & Chaysiri 2022) | `configs/model.yaml` | final |
| Sensitivitas jendela | `w` {30, 60, 120} dan `tau` {1, 5, 21} disetel pada periode desain | Menjadikan `w` dan `tau` hasil uji, bukan asumsi; hasil dilaporkan sebagai lampiran | `configs/model.yaml` `sensitivity` | final |
| Alternatif arsitektur yang ditolak | Model hibrida berbasis attention (CNN-BiLSTM-Attention/ECA, BiLSTM-Transformer) | Menjadi arah literatur 2024-2025, tetapi memerlukan data lebih banyak daripada 43 saham x 8 tahun; lingkup penelitian adalah replikasi CNN-BiLSTM (Chaweewanchon & Chaysiri 2022). Dicatat sebagai pengembangan berikutnya | `internal/SKRIPSI.md` Bagian 4.9 | final |

## Portofolio dan Evaluasi

| Keputusan | Nilai | Dasar | Implementasi | Status |
|---|---|---|---|---|
| Ukuran portofolio akhir | `k` = 5, 7, 10 | Pertanyaan penelitian RQ2 | `configs/portfolio.yaml` | final |
| Batas bobot | `max_weight` = 35% | Tidak ada artikel yang menetapkannya; diuji sensitivitas pada tahap portofolio (25%, 35%, 50%, 100%) | `configs/portfolio.yaml` | final |
| Estimator kovarians | sample, ridge, Ledoit-Wolf, GMV (perbandingan, bukan asumsi) | DeMiguel et al. (2009); Ledoit & Wolf (2004) | `configs/portfolio.yaml` | final |
| Rebalancing | bulanan (21 hari) | Espiga-Fernandez et al. (2024) (rebalancing periodik lebih hemat biaya); Huang et al. (2024) | `configs/portfolio.yaml` | final |
| Metrik | Sharpe, Sortino, Calmar, MDD, turnover, DSR, PBO | Bailey & Lopez de Prado (2014); Bailey et al. (2016) | `configs/experiment.yaml` | final |
| Uji signifikansi | Ledoit-Wolf HAC untuk Sharpe; Romano-Wolf untuk multiple testing | Ledoit & Wolf (2008) mengokohkan uji Jobson & Korkie (1981) terhadap non-normalitas dan dependensi deret waktu; Romano & Wolf (2005) untuk data snooping | `configs/experiment.yaml` | final |
| Rezim | COVID 2020; pemulihan/kenaikan suku bunga 2021-2025 (keduanya OOS) | Huang et al. (2024) (uji tahan COVID) | `configs/experiment.yaml` | final |
| Tingkat bebas risiko | BI-7DRRR harian (= tahunan / 252) | Bank Indonesia (sumber resmi); konversi sederhana oleh 252 hari | `configs/experiment.yaml` `evaluation.risk_free` | final |
| Faktor annualisasi | 252 | Konvensi standar. IDX empiris ~242 hari/tahun; diuji sebagai sensitivitas | `configs/experiment.yaml` `evaluation.annualization` | final |
| Kendala MV | long-only, jumlah bobot = 1 | Markowitz (1952); perdagangan ritel IDX tidak memperbolehkan short | `configs/portfolio.yaml` `constraints` | final |
| Regularisasi kovarians | ridge `epsilon` = 1e-4 | Ledoit & Wolf (2004) penyusutan untuk kovarians berdimensi besar; ridge menstabilkan diagonal | `configs/portfolio.yaml` `ridge_epsilon` | final |
| Penalti turnover | 0 | Biaya transaksi disimulasikan eksplisit (beli 0,19%, jual 0,29%, lot 100), bukan lewat penalti | `configs/portfolio.yaml` `turnover_penalty` | final |
| Model pembanding | MV klasik, Ridge, Random Forest, Gradient Boosting, CNN-BiLSTM; baseline 1/N dan IHSG | Chaweewanchon & Chaysiri (2022) membandingkan LSTM/BiLSTM/CNN-BiLSTM/MV; DeMiguel et al. (2009) untuk 1/N; Anuno & Madaleno (2024) untuk MVO klasik | `configs/experiment.yaml` `ablation` | final |
| Grid `k` | 5, 7, 10 (RQ2) | Chaweewanchon & Chaysiri (2022) menguji N = 5-10; Paiva et al. (2019) = 7; Wang et al. (2020) = 10 | `configs/portfolio.yaml` `k_values` | final |
| Uji Sharpe | Ledoit-Wolf HAC | Ledoit & Wolf (2008) | `configs/experiment.yaml` `significance.sharpe_test` | final |
| Multiple testing | Romano-Wolf stepdown | Romano & Wolf (2005); Bailey et al. (2016) untuk risiko backtest overfitting | `configs/experiment.yaml` `significance.multiple_testing` | final |
| Sensitivitas angka mandiri | `w` {30,60,120}, `tau` {1,5,21}, `L` {60,120,252}, `maxw` {25,35,50,100}%, `k` {3,15,20} | Mengubah angka tanpa artikel menjadi hasil uji; dilaporkan sebagai lampiran | `configs/model.yaml` `sensitivity`; `configs/portfolio.yaml` | final |
| Alternatif validasi yang ditolak | Combinatorial Purged Cross-Validation (CPCV) | Menjadi praktik baku 2024-2025, tetapi keluarga metodenya sudah diwakili PBO/CSCV (Bailey et al. 2016); walk-forward bersarang dengan purge/embargo (Lopez de Prado 2018) sudah dipakai | `internal/SKRIPSI.md` Bagian 4.9 | final |
| Formulasi Mean-Variance | Target-return constrained: min w'Σw s.t. w'μ=γ, Σw=1, 0≤w≤maxw | **Chaweewanchon & Chaysiri (2022) Section 3.1 Persamaan (1)-(4)** — formulasi eksplisit MV dengan expected return dari prediksi ML; Wang et al. (2020) LSTM+MV pakai prediksi sebagai expected return | `src/lq45/portfolio/optimize.py` `bobot_mean_varians_target_return` | final |
| Expected return (μ) untuk MV | Rata-rata `pred_ens` ensemble (5 seed) dari top-k saham terpilih | Chaweewanchon & Chaysiri (2022) "predicted results are integrated into the MV model"; Wang et al. (2020) LSTM+MV pakai prediksi sebagai expected return | `src/lq45/portfolio/backtest.py` `_target_return_dari_prediksi` | final |
| Target return (γ) | Mean pred_ens dari k saham terpilih pada tanggal rebalancing | Natural choice, no extra hyperparameter; konsisten dengan Ei pada persamaan Chaweewanchon & Chaysiri (2022) | `configs/portfolio.yaml` `optimization.target_return_method` | final |
| Grid optimizer type | `target_return` (MV) saja; GMV dihapus dari grid | Huang et al. (2024) Table 1 Panel B membandingkan AGC-CNN+GMV vs AGC-CNN+MaxSR vs AGC-CNN+1/N; dipilih MV karena menggunakan prediksi ML sebagai expected return | `configs/portfolio.yaml` `optimization.types` | final |
| Risk aversion λ | Tidak dipakai (formulasi target-return) | Formulasi Chaweewanchon & Chaysiri (2022) Eq 1-4 tidak menggunakan λ | - | final |
| Optimizer type dalam run_info | Kolom tambahan di output untuk audit | Traceability keputusan desain | `scripts/04_optimize.py` `run_info.json` | final |

---

## Implementasi Tahap 3

| Keputusan | Nilai | Dasar | Implementasi | Status |
|---|---|---|---|---|
| **Pre-training Self-Supervised (BARU)** | Masked Autoencoder (MAE) pada 7 channel (OHLCV + BI-7DRRR + JISDOR), mask_ratio=0.3, reconstruct OHLCV only | Kang (2025): LSTM on raw OHLCV ≥ technical indicators untuk prediksi harga; PatchTST (Nie et al. 2022), TimeMAE (Cheng et al. 2023), MTSMAE (Tang & Zhang 2022): MAE pre-train pada time series meningkatkan downstream forecasting | `configs/model.yaml` `pretrain`; `src/lq45/models/pretrain.py`; `scripts/03_train_predict.py` mode `pretrain` | final |
| **Fine-tune dengan Discriminative LR (BARU)** | Head LR=1e-4, Encoder LR=1e-5 (10x lebih kecil), freeze_encoder=false (full unfreeze) | Transfer learning standard practice; pre-trained encoder sudah belajar representasi universal, hanya head butuh adaptasi cepat | `configs/model.yaml` `pretrain.fine_tune`; `src/lq45/models/training.py` `fine_tune_model`; `scripts/03_train_predict.py` mode `walk-forward-pretrain` | final |
| **Input Pre-train** | 7 channel: Open, High, Low, Close, Volume, BI-7DRRR, JISDOR (tanpa indikator teknikal) | Kang (2025): LSTM on raw OHLCV alone matches XGBoost with 20+ technical indicators; indikator teknikal = deterministic transform of OHLCV → redundant untuk DL | `configs/model.yaml` `pretrain.input_channels=7`; `src/lq45/models/dataset.py` `PRETRAIN_CHANNELS` | final |
| **Pre-train Target** | Rekonstruksi OHLCV (5 channel) saja; makro sebagai conditioning | Makro (BI-7DRRR, JISDOR) slow-moving, exogenous; pola harga (candles, gaps, volume spikes) lebih kaya informasi untuk reconstruct | `src/lq45/models/decoder.py` `n_channels=5`; `src/lq45/models/pretrain.py` `mae_loss` pada channel :5 | final |
| **Pre-train Split** | Train 90% / Val 10% per saham (tanpa purge/embargo karena tidak ada label) | Pre-training unlabeled → tidak ada leakage; split per saham menghormati kalender independen | `src/lq45/models/walkforward.py` `make_pretrain_split`; `scripts/03_train_predict.py` `run_pretrain` | final |

## Implementasi Tahap 3

| Keputusan | Nilai | Dasar | Implementasi | Status |
|---|---|---|---|---|
| Pilihan CNN yang tidak ditetapkan artikel | Padding `same`, urutan Conv -> BatchNorm -> ReLU, inisialisasi bawaan PyTorch | Artikel tidak mematok detail ini; dicatat eksplisit agar dapat diaudit | `src/lq45/models/cnn_bilstm.py` | final |
| Titik fit praproses | Fit pada baris tanggal latih seluruh saham digabung (bukan pada elemen jendela); transformasi bekerja per elemen sehingga hasilnya identik | Konsisten dengan prinsip titik fit praproses (hanya data latih) dan model bersama seluruh saham | `scripts/03_train_predict.py` `skala_panel` | final |
| Pemeriksaan silang target | Target dihitung ulang dari `close`; saat tau=5 dicocokkan dengan kolom `target` CSV (`allclose`) | Mencegah pergeseran pipa tahap 2 yang tidak terdeteksi | `src/lq45/models/dataset.py` `load_panel` | final |
| Purge dan embargo mengikuti w/tau terpilih | Purge = tau, embargo = w-1 (bukan nilai baku 5/59 bila hasil tuning berbeda) | Konsisten dengan definisi di Lopez de Prado (2018) Bab 7 | `scripts/03_train_predict.py` pemanggilan `make_folds` | final |
| Dedup prediksi | Satu baris per (tanggal, saham, seed) dari lipatan terkecil yang memuat tanggal itu | Validasi lipatan berikutnya tumpang tindih dengan test lipatan sebelumnya (langkah 21 < test 21); lipatan terkecil = kemunculan pertama tanggal di evaluasi | `scripts/03_train_predict.py` `gabungkan` | final |
| Prediksi validasi pembuka OOS | 63 hari validasi pertama disimpan (sebelum test pertama) untuk tahap 4 tanpa lubang tanggal | Tahap 4 membutuhkan prediksi tiap tanggal rebalancing OOS | `scripts/03_train_predict.py` `gabungkan` | final |
| Determinisme | `build_model` menanam seed sebelum inisialisasi; dua run dengan seed sama identik bitwise (terverifikasi dengan uji) | Reproducibility; sebaran lintas seed dilaporkan (Reimers & Gurevych 2017) | `src/lq45/models/training.py`; `tests/test_determinism.py` | final |
| Pelanjutan dan paralelisme | Pelanjutan eksekusi terputus per (lipatan, seed); penulisan berkas atomik; paralel proses dengan thread per pekerja | Eksekusi panjang dapat dilanjutkan dan memakai 8 core tanpa merusak determinisme | `scripts/03_train_predict.py` | final |
| Data hilang karena suspensi | Jendela ber-NaN dibuang; saham tanpa prediksi pada tanggal itu dikecualikan dari peringkat | WIKA tersuspensi 2024-01-09 s.d. 2024-04-05 dan 2025-03-07 s.d. akhir periode; SRIL dan WSKT tidak tersedia di Yahoo Finance | `src/lq45/models/dataset.py` `build_windows`; `scripts/03_train_predict.py` `muat_panel` | final |

---

## Gerbang Penilai Ranker

| Keputusan | Nilai | Dasar | Implementasi | Status |
|---|---|---|---|---|
| Model dinilai sebagai ranker | IC peringkat dan spread top-minus-bottom, bukan MSE/R2 | Keputusan portofolio tahap 4 hanya memakai urutan; penyusutan skala mempertahankan urutan tetapi merusak MSE | `scripts/05_evaluate.py` | final |
| Peringkat ensemble lintas seed | Rata-rata pred_raw lima seed; sebaran dilaporkan | Meredam variansi pelatihan (Reimers & Gurevych 2017; Bouthillier et al. 2021) | `src/lq45/portfolio/ranking.py` `ensemble_prediksi` | final |
| Evaluasi utama memakai role=test | Baris validasi hanya untuk early stopping, tidak dinilai | Mencegah optimisme pemilihan model masuk ke hasil | `scripts/04_optimize.py` `kerangka_prediksi` | final |

---

## Implementasi Tahap 4-5

| Keputusan | Nilai | Dasar | Implementasi | Status |
|---|---|---|---|---|
| Kovarians dari data sampai tanggal rebalancing | Jendela `(d-L, d]` imbal hasil log Adj Close | Prinsip: mencegah kebocoran informasi ke depan | `src/lq45/portfolio/covariance.py` | final |
| Label gmv memakai kovarians sampel | Sama dengan sample pada kendala yang sama | Huang et al. (2024) memakai Global Minimum Variance tahap kedua; dipertahankan sebagai pembanding bernama | `src/lq45/portfolio/covariance.py` | final |
| Optimasi varians minimum SLSQP | Long-only, jumlah satu, batas bobot; gagal berarti sama rata | Markowitz (1952); kendala ritel IDX (long-only, sum=1) | `src/lq45/portfolio/optimize.py` | final |
| Optimasi mean-variance target-return | min w'Σw s.t. w'μ=γ; fallback ke GMV jika infeasible | Chaweewanchon & Chaysiri (2022) Section 3.1 Persamaan (1)-(4) | `src/lq45/portfolio/optimize.py` `bobot_mean_varians_target_return` | final |
| Expected returns passed to optimizer | Series `pred_ens` dari prediksi ensemble per tanggal rebalancing | Wang et al. (2020); Chaweewanchon & Chaysiri (2022) | `src/lq45/portfolio/backtest.py` | final |
| Pembulatan lot sebelum fee | Target lembar dibulatkan ke bawah ke kelipatan 100; fee dihitung dari selisih posisi | Ketentuan sekuritas ritel Indonesia (beli 0,19%, jual 0,29%, lot 100) | `src/lq45/portfolio/costs.py` | final |
| Modal awal 100 juta rupiah | Nilai bawaan `--modal`; dapat diubah | Mendekati skala ritel agar efek lot realistis; bukan sitasi | `scripts/04_optimize.py` | final |
| Jadwal rebalancing dari kalender test | Setiap 21 tanggal prediksi test | Selaras jendela test 21 hari tahap 3 | `src/lq45/portfolio/backtest.py` | final |
| Turnover-adjusted Sharpe | Sharpe dikali (1 - turnover); turnover = rerata \|selisih bobot\|/2 per rebalancing | Definisi implementasi; return sudah bersih dari fee sehingga penalti ganda dihindari | `scripts/05_evaluate.py` | final |
| DSR memakai seluruh konfigurasi sebagai jumlah uji | n_uji = jumlah baris metrik | Bailey & Lopez de Prado (2014): koreksi banyak percobaan | `scripts/05_evaluate.py` | final |

---

## Rujukan artikel

Semua entri bertanda **(OA)** adalah akses terbuka dan dapat diunduh langsung.

- Anuno, D. C., & Madaleno, M. (2024). Testing Portfolio Optimization in Timor-Leste. *JRFM* (MDPI). **(OA)**
- Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*. PDF: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf **(OA)**
- Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2016). The Probability of Backtest Overfitting. *Journal of Computational Finance*. PDF: https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf **(OA)**
- Bouthillier, X., Delaunay, P., Bronzi, M., et al. (2021). Accounting for Variance in Machine Learning Benchmarks. *MLSys*. https://arxiv.org/abs/2103.03098 **(OA)**
- Chaweewanchon, A., & Chaysiri, R. (2022). Markowitz Mean-Variance Portfolio Optimization with Predictive Stock Selection Using Machine Learning. *IJFS*, 10(3), 64. https://doi.org/10.3390/ijfs10030064 **(OA)**
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification. *Review of Financial Studies*. Versi kerja: https://users.nber.org/~confer/2006/si2006/ap/uppal.pdf **(OA)**
- Espiga-Fernandez, F., Garcia-Sanchez, A., & Ordieres-Mere, J. (2024). Systematic Portfolio Optimization. *Algorithms*, 17(12), 570. https://doi.org/10.3390/a17120570 **(OA)**
- Graves, A., Mohamed, A.-R., & Hinton, G. (2013). Speech Recognition with Deep Recurrent Neural Networks. *ICASSP*, 6645-6649. PDF: https://www.cs.toronto.edu/~graves/icassp_2013.pdf **(OA)**
- Huang, Y., et al. (2024). Enhancing Portfolio Optimization with Two-Stage Deep Learning. *Mathematics* (MDPI). **(OA)**
- Kim, S., et al. (2025). Robust Portfolio Optimization via Supervised Deep Ensembles. arXiv:2503.13544. https://arxiv.org/abs/2503.13544 **(OA)**
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411. Versi kerja: http://www.ledoit.net/honey.pdf **(OA)**
- Ledoit, O., & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *Journal of Empirical Finance*, 15(5), 850-859. PDF: http://www.ledoit.net/jef_2008pdf.pdf **(OA)**
- Malhotra, et al. (2023). Assessing Performance and Risk-Adjusted Returns. *IJFS* (MDPI). **(OA)**
- Reimers, N., & Gurevych, I. (2017). Reporting Score Distributions Makes a Difference. *EMNLP*. https://aclanthology.org/D17-1035 **(OA)**
- Romano, J. P., & Wolf, M. (2005). Stepwise Multiple Testing as Formalized Data Snooping. *Econometrica*, 73(4), 1237-1282. Versi kerja tersedia di Cowles Foundation. **(OA versi kerja)**
- Sebastian, T. A., & Tantia, R. (2024). Deep Learning Stock Price Prediction and Portfolio Optimization. *IJACSA*. https://thesai.org/Downloads/Volume15No9/Paper_95-Deep_Learning_for_Stock_Price_Prediction.pdf **(OA)**
- Sen, J., & Dutta, A. (2021). Stock Portfolio Optimization Using Deep Learning LSTM Model. arXiv:2111.04709. https://arxiv.org/abs/2111.04709 **(OA)**
- Wang, W., Li, W., Zhang, N., & Liu, K. (2020). Portfolio formation with preselection using deep learning from long-term financial data. *Expert Systems with Applications*. PDF repositori: https://centaur.reading.ac.uk/86775/3/portfolio%20formation_revised2_20191010.pdf **(OA)**
- Wang, X., & Liu, X. (2025). Risk-Sensitive Deep Reinforcement Learning for Portfolio Optimization. *JRFM* (MDPI). **(OA)**

**Baru untuk Pre-training (OA):**
- Kang, S. (2025). Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data: Evidence from Korean Markets. arXiv:2504.02249. https://arxiv.org/abs/2504.02249 **(OA)**
- Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2022). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST). arXiv:2211.14730. https://arxiv.org/abs/2211.14730 **(OA)**
- Cheng, M., Tao, X., Liu, Z., Liu, Q., Zhang, H., Zhang, R., & Chen, E. (2023). TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders. arXiv:2303.00320. https://arxiv.org/abs/2303.00320 **(OA)**
- Tang, P., & Zhang, X. (2022). MTSMAE: Masked Autoencoders for Multivariate Time-Series Forecasting. arXiv:2210.02199. https://arxiv.org/abs/2210.02199 **(OA)**

**Tidak akses terbuka** (buku dan artikel klasik; dikutip sebagai dasar, bukan untuk diunduh):

- Jobson, J. D., & Korkie, B. M. (1981). Performance Hypothesis Testing with the Sharpe and Treynor Measures. *Journal of Finance*, 36(4), 889-908. Uji asli yang diperbaiki oleh Ledoit & Wolf (2008).
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Dasar purging dan embargo (Bab 7).
- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77-91.
- Paiva, F. D., et al. (2019). Decision-making for financial trading. *Expert Systems with Applications*. Dikutip melalui Chaweewanchon & Chaysiri (2022).

Catatan: salinan PDF artikel berada di `Literature/` (internal, tidak
dipublikasikan karena hak cipta). Berkas ini hanya memuat sitasi, bukan isi.

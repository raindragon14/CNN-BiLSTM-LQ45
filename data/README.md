# Data

> **Peran:** sumber data, definisi fitur, dan keterbatasan.
> **Audiens:** publik.
> **Bukan untuk:** dasar keputusan (lihat `docs/keputusan_desain.md`) atau urutan
> skrip (lihat `scripts/README.md`).

Data penelitian tidak disertakan dalam repositori (ukuran berkas dan ketentuan sumber).

## Struktur

- `raw/`: hasil unduhan mentah, tidak diubah
  - `prices/*.csv`: OHLCV harian per saham (kolom `Close` dan `Adj Close`)
  - `macro/jisdor.csv`: kurs referensi resmi USD/IDR (BI)
  - `macro/bi_7drrr.csv`: riwayat keputusan suku bunga acuan BI
  - `metadata.json`: waktu unduh, daftar ticker, jumlah baris, ticker tanpa data
- `interim/`: hasil pembersihan (kalender bursa, forward-fill maks. 3 hari, log-return)
- `processed/features/`: 8 fitur + target siap model (lihat bagian Fitur dan target)

## Sumber dan justifikasi

| Data | Sumber | Sifat | Catatan |
|---|---|---|---|
| Harga saham | Yahoo Finance (`.JK`), `adjusted close` | Sekunder | Disesuaikan split dan dividen; diungkap sebagai keterbatasan |
| Kurs USD/IDR | Bank Indonesia, JISDOR | Primer | Kurs referensi resmi; diunduh dari `bi.go.id` |
| Suku bunga acuan | Bank Indonesia, BI-7DRRR | Primer | Tanggal publikasi keputusan, bukan tanggal efektif |
| Tolok ukur | IHSG (`.JKSE`) | Sekunder | Untuk pembanding beli-dan-tahan |

Catatan keterpertanggungjawaban:

- Dua variabel makro berasal dari **Bank Indonesia** (sumber primer), bukan
  agregator pihak ketiga.
- Harga saham berasal dari Yahoo Finance. Sumber ini bukan sumber primer,
  sehingga metode penyesuaian harga (split/dividen) tidak sepenuhnya
  transparan. Keterbatasan ini harus dinyatakan pada laporan; validasi silang
  terhadap sumber resmi IDX disarankan sebelum hasil final dilaporkan.
- Tanggal makro memakai **tanggal publikasi** untuk mencegah kebocoran
  informasi ke depan.
- Nilai JISDOR harian dipakai pada tanggal yang sama (tanpa lag tambahan).
  JISDOR terbit pada sore hari sehingga pemakaian seketika bersifat
  sezaman; dampaknya terbatas karena hanya satu dari delapan fitur dan
  dicatat di sini sebagai keterbatasan.

## Pool kandidat

Pool dibaca dari `configs/universe.yaml`:

- **Komposisi:** 45 konstituen LQ45 yang berlaku pada periode Agustus 2019 -
  Januari 2020.
- **Sumber:** IDX LQ45 Company Profiles, Agustus 2019 (halaman *Contents*).
  Dokumen arsip dapat diverifikasi pada Wayback Machine.
- **Alasan:** komposisi ini berlaku tepat sebelum periode out-of-sample dimulai
  (Januari 2020), sehingga tidak memakai informasi masa depan dan tidak
  menimbulkan bias *survivorship* dari daftar akhir periode. Pool kemudian
  dibekukan untuk seluruh periode penelitian.
- **Ketersediaan:** 43 dari 45 ticker tersedia di Yahoo Finance. `SRIL.JK` dan
  `WSKT.JK` tidak tersedia karena suspensi/delisting.

Keterbatasan yang harus dinyatakan:

- Pool dibekukan, bukan *point-in-time*. Saham yang masuk atau keluar LQ45
  setelah Januari 2020 tidak diperlakukan secara dinamis.
- Dua saham yang tersuspensi tidak dapat diikutsertakan karena keterbatasan
  data harga.

## Fitur dan target

Delapan fitur per saham. Spesifikasi: `configs/experiment.yaml`; jejak
keputusan: `docs/keputusan_desain.md`.

| Fitur | Nilai | Sumber |
|---|---|---|
| `close` | `Adj Close` level | Sebastian & Tantia (2024); Sen & Dutta (2021) |
| `volume` | Volume harian | Sebastian & Tantia (2024); Sen & Dutta (2021) |
| `rsi` | RSI periode 14 | Espiga-Fernandez et al. (2024) Lampiran B.3 |
| `cci` | CCI periode 20 | Espiga-Fernandez et al. (2024) Lampiran B.4 |
| `cmo` | CMO periode 14 | Espiga-Fernandez et al. (2024) Lampiran B.5 |
| `mfi` | MFI periode 14 | Espiga-Fernandez et al. (2024) Lampiran B.6 |
| `bi_7drrr` | BI-7DRRR level | Ekstensi (celah Chaweewanchon & Chaysiri 2022) |
| `jisdor` | Log-return USD/IDR | Ekstensi (idem) |

- **Target**: log-return 5 hari ke depan.
- **Praproses**: winsorization `[Q1 - 1,5*IQR, Q3 + 1,5*IQR]` lalu min-max,
  diterapkan pada fitur dan target, di-*fit* **hanya pada data latih** tiap
  jendela walk-forward (mencegah kebocoran informasi).
- AROONOSC, Williams %R, dan STOCHF dari Espiga-Fernandez et al. (2024) tidak
  diambil agar jumlah fitur tetap 8. Penyederhanaan ini disengaja.

## Biaya transaksi

- Fee beli 0,19%, fee jual 0,29%, lot 100 lembar (ketentuan sekuritas ritel).
- Rebalancing bulanan (21 hari bursa).

"""Pengambilan data mentah dari sumber eksternal.

Sumber:
- Harga saham ``.JK``: Yahoo Finance.
- Kurs USD/IDR (JISDOR) dan BI-7DRRR: Bank Indonesia.

Fungsi di sini mengembalikan data apa adanya; pembersihan dikerjakan pada
tahap berikutnya (``02_build_features.py``).
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from collections.abc import Sequence
from xml.etree import ElementTree

import pandas as pd
import requests

BI_RATE_URL = "https://www.bi.go.id/en/statistik/indikator/bi-rate.aspx"
BI_JISDOR_URL = "https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/default.aspx"
_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124 Safari/537.36"
)


def _session() -> requests.Session:
    """Buat sesi HTTP dengan User-Agent peramban."""
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    return session


def _request(session: requests.Session, method: str, **kwargs) -> requests.Response:
    """Permintaan HTTP dengan percobaan ulang sederhana."""
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            return getattr(session, method)(**kwargs)
        except requests.RequestException as exc:  # koneksi situs BI kerap putus
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"permintaan gagal setelah 4 percobaan: {last_error}")


def fetch_equities(
    tickers: Sequence[str],
    start: str,
    end: str,
    retries: int = 3,
) -> dict[str, pd.DataFrame]:
    """Unduh OHLCV harian per ticker melalui Yahoo Finance.

    ``auto_adjust=False`` agar ``Adj Close`` terpisah dari ``Close``. Ticker
    tanpa data (misalnya delisting) dipetakan ke ``None``.
    """
    import yfinance as yf

    hasil: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame: pd.DataFrame | None = None
        for attempt in range(retries):
            try:
                frame = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2)
        if frame is None or frame.empty:
            hasil[ticker] = None
            continue
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.index.name = "Date"
        hasil[ticker] = frame
        time.sleep(0.4)
    return hasil


def _xlsx_rows(content: bytes) -> list[dict[str, str]]:
    """Baca baris lembar pertama ``.xlsx`` tanpa pustaka tambahan.

    Hanya menangani ekspor satu lembar dari Bank Indonesia: teks bersama dan
    angka langsung, tanpa rumus.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as berkas:
        bersama: list[str] = []
        if "xl/sharedStrings.xml" in berkas.namelist():
            akar = ElementTree.fromstring(berkas.read("xl/sharedStrings.xml"))
            for si in akar.findall(f"{_XLSX_NS}si"):
                bersama.append("".join(t.text or "" for t in si.iter(f"{_XLSX_NS}t")))
        lembar = ElementTree.fromstring(berkas.read("xl/worksheets/sheet1.xml"))

    baris: list[dict[str, str]] = []
    for row in lembar.iter(f"{_XLSX_NS}row"):
        sel: dict[str, str] = {}
        for c in row.findall(f"{_XLSX_NS}c"):
            ref = c.get("r", "")
            kolom = re.match(r"[A-Z]+", ref).group(0)
            nilai = c.find(f"{_XLSX_NS}v")
            if c.get("t") == "s" and nilai is not None:
                sel[kolom] = bersama[int(nilai.text)]
            else:
                sel[kolom] = nilai.text if nilai is not None else ""
        baris.append(sel)
    return baris


def fetch_jisdor(start: str, end: str) -> pd.DataFrame:
    """Unduh kurs referensi resmi JISDOR (USD/IDR) dari Bank Indonesia.

    Situs BI menyediakan pemilih rentang tanggal dan tombol ekspor; fungsi ini
    menirukan permintaan tersebut lalu membaca berkas ``.xlsx`` hasilnya.
    Kolom keluaran: ``date`` dan ``rate`` (rupiah per 1 USD).
    """
    session = _session()
    html = _request(session, "get", url=BI_JISDOR_URL, timeout=25).text

    awalan = re.search(r'name="([^"]*TextBoxFrom)"', html)
    if awalan is None:
        raise RuntimeError("kontrol rentang tanggal JISDOR tidak ditemukan")
    prefix = awalan.group(1).rsplit("TextBoxFrom", 1)[0]

    awal = pd.Timestamp(start).strftime("%d/%m/%Y")
    akhir = pd.Timestamp(end).strftime("%d/%m/%Y")
    muatan = _bi_hidden(html)
    muatan[prefix + "TextBoxFrom"] = awal
    muatan[prefix + "TextBoxDateTo"] = akhir
    muatan[prefix + "HiddenFieldDateFrom"] = awal
    muatan[prefix + "HiddenFieldDateTo"] = akhir
    muatan[prefix + "ButtonExport"] = "Unduh"

    respon = _request(session, "post", url=BI_JISDOR_URL, data=muatan, timeout=120)
    if respon.content[:2] != b"PK":
        raise RuntimeError("ekspor JISDOR bukan berkas .xlsx")

    catatan: list[dict[str, object]] = []
    for sel in _xlsx_rows(respon.content):
        tanggal = sel.get("B", "")
        kurs = sel.get("C", "")
        if not tanggal or not kurs:
            continue
        waktu = pd.to_datetime(tanggal, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        if pd.isna(waktu):
            continue
        try:
            nilai = float(kurs)
        except ValueError:
            continue
        catatan.append({"date": waktu, "rate": nilai})
    hasil = pd.DataFrame(catatan).dropna(subset=["date"])
    return hasil.sort_values("date").reset_index(drop=True)


def _bi_table(html: str) -> pd.DataFrame:
    """Baca tabel pertama dari HTML halaman Bank Indonesia."""
    tabel = pd.read_html(io.StringIO(html))
    frame = tabel[0]
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _bi_next_target(html: str) -> str | None:
    """Nama tombol "Next" pada DataPager bila masih aktif."""
    for match in re.finditer(r"<input type=\"image\"[^>]*>", html):
        tag = match.group(0)
        if "DataPagerBI7DRR" in tag and "next" in tag and "disabled" not in tag:
            return re.search(r'name="([^"]+)"', tag).group(1)
    return None


def _bi_hidden(html: str) -> dict[str, str]:
    """Kumpulkan seluruh input tersembunyi pada formulir ASP.NET."""
    return {
        match.group(1): match.group(2)
        for match in re.finditer(
            r'<input type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', html
        )
    }


def fetch_bi_rate(max_pages: int = 40) -> pd.DataFrame:
    """Ambil seluruh riwayat keputusan BI-7DRRR (suku bunga acuan).

    Halaman indikator menampilkan 10 pertemuan per halaman; fungsi ini
    menelusuri seluruh halaman lalu mengembalikan kolom ``date`` dan ``rate``
    (dalam persen).
    """
    session = _session()
    response = _request(session, "get", url=BI_RATE_URL, timeout=25)
    html = response.text

    catatan: dict[int, tuple[str, float]] = {}
    for _ in range(max_pages):
        frame = _bi_table(html)
        for _, baris in frame.iterrows():
            catatan[int(baris.iloc[0])] = (
                str(baris.iloc[1]),
                float(str(baris.iloc[2]).replace("%", "").strip()),
            )
        target = _bi_next_target(html)
        if target is None:
            break
        payload = _bi_hidden(html)
        payload[f"{target}.x"] = "1"
        payload[f"{target}.y"] = "1"
        payload["__EVENTTARGET"] = ""
        payload["__EVENTARGUMENT"] = ""
        html = _request(session, "post", url=BI_RATE_URL, data=payload, timeout=30).text
        time.sleep(0.6)

    if not catatan:
        raise RuntimeError("tabel BI-7DRRR tidak terbaca")

    # Nomor urut menurun terhadap waktu; urutkan naik menurut tanggal.
    baris = [
        {
            "date": pd.to_datetime(periode, format="%d %B %Y"),
            "rate": rate,
        }
        for periode, rate in catatan.values()
    ]
    hasil = pd.DataFrame(baris).sort_values("date").reset_index(drop=True)
    return hasil

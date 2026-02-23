# Checkee.info Visa Statistics Visualizer

An interactive Streamlit dashboard that scrapes [checkee.info](https://www.checkee.info/) and visualizes US visa Administrative Processing (AP) case statistics with Plotly charts.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.32%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

| Tab | What's inside |
|-----|---------------|
| **Overview** | Status pie chart, cases by visa type / consulate / major, monthly trend |
| **Location Comparison** | Violin plots, ECDF curves, clear-rate bar, dual heatmaps (median wait & clear rate %) — helps you pick the fastest consulate |
| **Major & Degree** | Auto-detected degree level (PhD / Master's / MBA / Bachelor's), waiting-time violins by consulate × degree, ECDF by degree |
| **Waiting Time Deep Dive** | Percentile table, histogram, box plot by visa type, scatter of check date vs wait days with LOWESS trend |
| **Raw Data** | Filterable table + CSV download |

**Sidebar controls**

- Date range picker (2008 – present)
- Global filters: Visa Type, Consulate, Entry Type
- Per-tab local filters: Major keyword search, Degree level
- **Test Connection** button — tells you instantly if the server is reachable
- **Use Sample Data** button — loads synthetic data for UI preview when the real site is rate-limited

**Smart caching**

Scraped months are saved to `checkee_cache.json` on disk. Re-loading the same range is instant; only new or the current month hits the server again.

---

## Installation

```bash
git clone https://github.com/<your-username>/checkee-viz.git
cd checkee-viz
pip install -r requirements.txt
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Requirements

- Python 3.9+
- See [`requirements.txt`](requirements.txt) for the full list

```
streamlit
pandas
plotly
requests
beautifulsoup4
lxml
matplotlib
statsmodels
numpy
```

---

## Usage

1. Select a **date range** in the sidebar (default: last 3 months for a fast first load).
2. Click **Load / Refresh Data**.
3. Use the **Global Filters** (sidebar) and per-tab filters to slice the data.
4. Switch between tabs to explore different views.

### Rate limiting

checkee.info limits how frequently it can be scraped. If you see a rate-limit warning:

- Click **Test Connection** periodically — it uses a single request to check.
- Once it shows ✅ Connected, click **Load / Refresh Data**.
- Past months are cached to disk and won't be re-fetched.
- Click **Use Sample Data** to explore the full UI with synthetic data in the meantime.

---

## Data fields

Each record scraped from checkee.info contains:

| Field | Example |
|-------|---------|
| Month | `2025-12` |
| Visa Type | `F1`, `H1`, `J1`, `B1`, … |
| Entry | `New` / `Renewal` |
| Consulate | `BeiJing`, `ShangHai`, `GuangZhou`, … |
| Major | `Computer Science`, `EE/BME MS`, `CS PhD`, … |
| **Degree** *(derived)* | `PhD`, `Master's`, `MBA`, `Bachelor's`, `Not Specified` |
| Status | `Pending`, `Clear`, `Reject` |
| Check Date | `2025-12-01` |
| Complete Date | `2025-12-15` (or blank if pending) |
| Waiting Days | `14` |

---

## Screenshots

> Load the app and click **Use Sample Data** to see all charts immediately.

---

## License

MIT — see [LICENSE](LICENSE).

> **Disclaimer:** This tool scrapes a third-party website. Use responsibly and respect the site's rate limits. The authors are not affiliated with checkee.info.

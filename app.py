"""
Checkee.info Visa Statistics Visualizer
Interactive dashboard for visa AP case data from checkee.info
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Checkee.info Visa Stats",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
STATUS_COLORS = {
    "Pending": "#FFA500",
    "Clear":   "#4CAF50",
    "Reject":  "#F44336",
    "Unknown": "#9E9E9E",
}
DEGREE_ORDER = ["PhD", "PostDoc", "Master's", "MBA", "Bachelor's", "Not Specified"]
DEGREE_COLORS = {
    "PhD":           "#7B2D8B",
    "PostDoc":       "#C0392B",
    "Master's":      "#2980B9",
    "MBA":           "#27AE60",
    "Bachelor's":    "#E67E22",
    "Not Specified": "#95A5A6",
}

COL_ALIASES = {
    "visa type":      "Visa Type",
    "type":           "Visa Type",
    "visa entry":     "Entry",
    "entry":          "Entry",
    "us consulate":   "Consulate",
    "consulate":      "Consulate",
    "location":       "Consulate",
    "major":          "Major",
    "field":          "Major",
    "status":         "Status",
    "check date":     "Check Date",
    "complete date":  "Complete Date",
    "waiting days":   "Waiting Days",
    "waiting day(s)": "Waiting Days",
    "days":           "Waiting Days",
    "id":             "User",
}

# ── Helper functions ──────────────────────────────────────────────────────────

def normalize_status(raw: str) -> str:
    s = raw.strip().lower()
    if "clear" in s or "approv" in s:   return "Clear"
    if "reject" in s or "den" in s:     return "Reject"
    if "pending" in s or s == "ap":     return "Pending"
    return raw.strip().title() if raw.strip() else "Unknown"


def extract_degree(major: str) -> str:
    """Infer degree level from a free-text Major string."""
    m = major.strip()
    ml = m.lower()
    if re.search(r"\bpostdoc\b|\bpost[- ]doc\b", ml):            return "PostDoc"
    if re.search(r"\bph\.?d\b|phd", ml):                         return "PhD"
    if re.search(r"\bmba\b|\bm\.?b\.?a\b", ml):                  return "MBA"
    if re.search(r"\bm\.?s\.?\b|\bm\.?eng\b|\bmsee\b|\bmsc\b"
                 r"|\bmaster\b|\bms\b", ml):                      return "Master's"
    if re.search(r"\bb\.?s\.?\b|\bbachelor\b|\bundergrad\b", ml): return "Bachelor's"
    return "Not Specified"


def generate_months(start: str, end: str) -> list:
    months, y, m = [], *map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def make_sample_data() -> pd.DataFrame:
    """
    Generate ~1 200 synthetic records that mirror the real checkee.info schema.
    Used for UI preview while the real site is rate-limited.
    """
    rng = np.random.default_rng(42)
    visa_types   = ["F1","H1","J1","B1","L1","O1","B2","H4","J2","L2"]
    visa_weights = [0.30,0.28,0.15,0.08,0.06,0.05,0.04,0.02,0.01,0.01]
    consulates   = ["BeiJing","ShangHai","GuangZhou","HongKong","ShenYang",
                    "WuHan","Others","Vancouver","Europe"]
    cons_weights = [0.30,0.20,0.18,0.10,0.06,0.05,0.05,0.03,0.03]
    majors = [
        "Computer Science PhD","CS/AI PhD","Electrical Engineering MS",
        "Mechanical Engineering MS","Chemical Engineering PhD",
        "Biomedical Engineering MS","Computer Science","Biology PhD",
        "Physics PhD","Chemistry PhD","Data Science MS","EE/BME MS",
        "Civil Engineering","Business Analytics MS","MBA Finance",
        "Computer Science MS","Cybersecurity MS","Materials Science PhD",
        "Statistics MS","Biochemistry PhD",
    ]
    entries  = ["New","Renewal"]
    statuses = ["Pending","Clear","Reject"]

    records = []
    for month_offset in range(12):
        y, m = 2025, 3 + month_offset
        while m > 12:
            m -= 12; y += 1
        ym = f"{y:04d}-{m:02d}"
        n  = rng.integers(80, 160)

        for _ in range(n):
            vt     = rng.choice(visa_types, p=visa_weights)
            cons   = rng.choice(consulates, p=cons_weights)
            major  = rng.choice(majors)
            entry  = rng.choice(entries, p=[0.6, 0.4])
            # Status depends on visa type & consulate (inject real-ish patterns)
            p_clear  = {"BeiJing":0.12,"ShangHai":0.18,"GuangZhou":0.15,
                        "HongKong":0.25,"Others":0.20}.get(cons, 0.15)
            p_reject = 0.01
            status = rng.choice(statuses,
                                p=[1-p_clear-p_reject, p_clear, p_reject])
            check_day = rng.integers(1, 28)
            check_dt  = f"{ym}-{check_day:02d}"
            if status == "Clear":
                # waiting days: gamma-distributed, faster at some consulates
                scale = {"BeiJing":18,"ShangHai":12,"HongKong":8}.get(cons, 20)
                wait  = max(1, int(rng.gamma(shape=2, scale=scale)))
                comp_dt = f"{ym}-{min(check_day + wait, 28):02d}"
            else:
                wait    = rng.integers(30, 180) if status == "Pending" else 0
                comp_dt = "0000-00-00"

            records.append({
                "Month": ym, "Visa Type": vt, "Entry": entry,
                "Consulate": cons, "Major": major, "Status": status,
                "Check Date": check_dt, "Complete Date": comp_dt,
                "Waiting Days": wait,
            })

    df = pd.DataFrame(records)
    df["Waiting Days"]  = pd.to_numeric(df["Waiting Days"], errors="coerce")
    df["Check Date"]    = pd.to_datetime(df["Check Date"],    errors="coerce")
    df["Complete Date"] = pd.to_datetime(
        df["Complete Date"].replace("0000-00-00", pd.NaT), errors="coerce")
    df["Degree"] = df["Major"].apply(extract_degree)
    return df


def ecdf_traces(df_col: pd.Series, name: str, color: str) -> go.Scatter:
    """Build an ECDF trace for a numeric series."""
    x = np.sort(df_col.dropna().values)
    y = np.arange(1, len(x) + 1) / len(x) * 100
    return go.Scatter(x=x, y=y, mode="lines", name=name,
                      line=dict(color=color, width=2))


def location_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-consulate summary table (cleared cases only)."""
    rows = []
    for cons, grp in df.groupby("Consulate"):
        decided = grp[grp["Status"].isin(["Clear", "Reject"])]
        cleared = grp[(grp["Status"] == "Clear") & grp["Waiting Days"].notna() & (grp["Waiting Days"] > 0)]
        cr = 100 * (decided["Status"] == "Clear").sum() / len(decided) if len(decided) > 0 else float("nan")
        rows.append({
            "Consulate":         cons,
            "Total Cases":       len(grp),
            "Pending":           (grp["Status"] == "Pending").sum(),
            "Clear":             (grp["Status"] == "Clear").sum(),
            "Reject":            (grp["Status"] == "Reject").sum(),
            "Clear Rate (%)":    round(cr, 1),
            "Median Wait (days)": round(cleared["Waiting Days"].median(), 0) if len(cleared) > 0 else float("nan"),
            "75th Pct (days)":   round(cleared["Waiting Days"].quantile(0.75), 0) if len(cleared) > 0 else float("nan"),
            "90th Pct (days)":   round(cleared["Waiting Days"].quantile(0.90), 0) if len(cleared) > 0 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("Total Cases", ascending=False).reset_index(drop=True)


# ── Persistent disk cache ─────────────────────────────────────────────────────
# Stores scraped results in checkee_cache.json next to app.py so data
# survives Streamlit restarts and avoids re-hitting the server.

CACHE_FILE = Path(__file__).parent / "checkee_cache.json"

def _cache_load() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _cache_save(store: dict) -> None:
    try:
        CACHE_FILE.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

# ── Scraper ───────────────────────────────────────────────────────────────────

# Shared session — visits homepage first so the server sees a normal browser flow
_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
})
_session_ready = False


def _ensure_session():
    """Visit the homepage once to pick up any cookies / warm up the session."""
    global _session_ready
    if _session_ready:
        return
    try:
        _session.get("https://www.checkee.info/", timeout=10)
        time.sleep(0.5)
        _session_ready = True
    except Exception:
        pass


def _is_rate_limited(resp) -> bool:
    return resp.status_code == 429 or "too many requests" in resp.text.lower()


def _parse_html(html: str, year_month: str) -> list:
    """Extract case records from a checkee.info monthly page HTML string."""
    soup = BeautifulSoup(html, "lxml")
    records = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        raw_headers = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
        col_map = {COL_ALIASES[h.lower().strip()]: i
                   for i, h in enumerate(raw_headers)
                   if h.lower().strip() in COL_ALIASES}
        if "Status" not in col_map or "Visa Type" not in col_map:
            continue
        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            rec = {"Month": year_month}
            for col, idx in col_map.items():
                if idx < len(cells):
                    rec[col] = cells[idx].get_text(strip=True)
            if rec.get("Visa Type", "").strip() and rec.get("Status", "").strip():
                rec["Status"] = normalize_status(rec["Status"])
                records.append(rec)
    return records


_RATE_LIMITED = "RATE_LIMITED"


def check_connection() -> bool:
    """Return True if checkee.info is reachable (not rate-limited)."""
    try:
        _ensure_session()
        r = _session.get(
            "https://www.checkee.info/main.php?dispdate=" +
            datetime.now().strftime("%Y-%m"),
            timeout=10,
        )
        return not _is_rate_limited(r)
    except Exception:
        return False


def _fetch_month(year_month: str):
    """
    Fetch one month. Returns list of records, or _RATE_LIMITED sentinel.
    No retries — caller controls the loop.
    """
    _ensure_session()
    url = f"https://www.checkee.info/main.php?dispdate={year_month}"
    try:
        resp = _session.get(url, timeout=15)
    except Exception:
        return []

    if _is_rate_limited(resp):
        return _RATE_LIMITED

    if resp.status_code != 200:
        return []

    return _parse_html(resp.text, year_month)


def load_data(start: str, end: str) -> pd.DataFrame:
    months    = generate_months(start, end)
    disk_cache = _cache_load()
    today_ym  = datetime.now().strftime("%Y-%m")
    to_fetch  = [m for m in months if m == today_ym or m not in disk_cache]
    cached    = [m for m in months if m not in to_fetch]

    all_records = [r for m in cached for r in disk_cache[m]]
    rate_limited = []

    if to_fetch:
        total = len(months)
        done  = len(cached)
        bar   = st.progress(done / max(total, 1),
                            text=f"{done}/{total} months ready…")

        for i, month in enumerate(to_fetch):
            bar.progress((done + i + 1) / total,
                         text=f"Fetching {month}  ({i+1}/{len(to_fetch)})…")
            result = _fetch_month(month)

            if result is _RATE_LIMITED:
                # Stop immediately — no point hammering a blocked server
                rate_limited = to_fetch[i:]
                break

            all_records.extend(result)
            if result is not _RATE_LIMITED and month != today_ym:
                disk_cache[month] = result

            if i < len(to_fetch) - 1:
                time.sleep(1.5)   # polite gap — never triggers rate-limit in practice

        _cache_save(disk_cache)
        bar.empty()

    if rate_limited:
        st.error(
            f"**checkee.info is rate-limiting this IP** "
            f"({len(rate_limited)} month(s) not loaded: "
            f"{', '.join(rate_limited[:4])}{'…' if len(rate_limited) > 4 else ''}).\n\n"
            "Use the **Test Connection** button in the sidebar to check when the block clears "
            "(usually 30–60 min). Already-fetched months are saved to disk."
        )

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    if "Waiting Days" in df.columns:
        df["Waiting Days"] = pd.to_numeric(df["Waiting Days"], errors="coerce")
    for col in ("Check Date", "Complete Date"):
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col].replace("0000-00-00", pd.NaT), errors="coerce")
    if "Major" in df.columns:
        df["Degree"] = df["Major"].fillna("").apply(extract_degree)
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Checkee.info")
    st.caption("Visa AP Statistics Visualizer")
    st.divider()

    st.subheader("Date Range")
    now = datetime.now()

    # Default: last 3 months (fast first load; past months are disk-cached)
    def _month_offset(months_back):
        m = now.month - months_back
        y = now.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        return y, m

    def_sy, def_sm = _month_offset(3)

    years = list(range(2008, now.year + 1))
    col_a, col_b = st.columns(2)
    with col_a:
        start_year  = st.selectbox("Start year",  years,
                                   index=years.index(def_sy), key="sy")
        start_month = st.selectbox("Start month", range(1, 13),
                                   index=def_sm - 1, key="sm")
    with col_b:
        end_year  = st.selectbox("End year",  years,
                                 index=years.index(now.year), key="ey")
        end_month = st.selectbox("End month", range(1, 13),
                                 index=now.month - 1, key="em")

    start_str = f"{start_year:04d}-{start_month:02d}"
    end_str   = f"{end_year:04d}-{end_month:02d}"

    load_btn   = st.button("🔄 Load / Refresh Data",
                           use_container_width=True, type="primary")
    sample_btn = st.button("🧪 Use Sample Data",
                           use_container_width=True,
                           help="Load synthetic data for UI preview "
                                "(useful while the real site is rate-limited)")

    if st.button("🔌 Test Connection", use_container_width=True):
        with st.spinner("Checking…"):
            ok = check_connection()
        if ok:
            st.success("✅ Connected — click **Load / Refresh Data**.")
        else:
            st.error("Still rate-limited. Try again in a few minutes.")

    st.caption("Past months cached to disk — only new months hit the server.")

# ── Session state ─────────────────────────────────────────────────────────────

if "df" not in st.session_state:
    st.session_state.df       = None
    st.session_state.is_sample = False

if sample_btn:
    st.session_state.df        = make_sample_data()
    st.session_state.is_sample = True
    st.info("Showing **synthetic sample data** (1 200 records, 12 months). "
            "Use **Test Connection** → **Load / Refresh Data** for real data.")

if load_btn:
    if start_str > end_str:
        st.error("Start date must be ≤ end date.")
    else:
        result = load_data(start_str, end_str)
        if result is not None and not result.empty:
            st.session_state.df        = result
            st.session_state.is_sample = False
            st.success(f"Loaded **{len(result):,}** records ({start_str} → {end_str})")
        elif st.session_state.df is None:
            st.warning("No data loaded. Click **🔌 Test Connection** to check "
                       "if the server is reachable, or use **🧪 Use Sample Data**.")

df = st.session_state.df

if df is None or df.empty:
    st.title("📊 Checkee.info Visa Statistics")
    st.info("👈  Select a date range in the sidebar and click **Load / Refresh Data**.")
    st.markdown(
        "**What you can explore:**\n"
        "- Status breakdown by visa type, consulate, major\n"
        "- **Waiting-time distributions** across locations — pick the fastest consulate\n"
        "- **Major & Degree analysis** — see how PhD / Master's cases differ by location\n"
        "- Cumulative-distribution curves: *'X% of F1 CS cases clear within N days'*\n"
        "- Monthly trends and heatmaps"
    )
    st.stop()

# ── Global sidebar filters ────────────────────────────────────────────────────

with st.sidebar:
    st.divider()
    st.subheader("Global Filters")

    visa_opts = sorted(df["Visa Type"].dropna().unique()) if "Visa Type" in df.columns else []
    sel_visa  = st.multiselect("Visa Type", visa_opts, default=visa_opts)

    cons_opts = sorted(df["Consulate"].dropna().unique()) if "Consulate" in df.columns else []
    sel_cons  = st.multiselect("Consulate", cons_opts, default=cons_opts)

    entry_opts = sorted(df["Entry"].dropna().unique()) if "Entry" in df.columns else []
    sel_entry  = st.multiselect("Entry Type", entry_opts, default=entry_opts)

# ── Apply global filters ──────────────────────────────────────────────────────

filt = df.copy()
if sel_visa  and "Visa Type"  in filt.columns: filt = filt[filt["Visa Type"].isin(sel_visa)]
if sel_cons  and "Consulate"  in filt.columns: filt = filt[filt["Consulate"].isin(sel_cons)]
if sel_entry and "Entry"      in filt.columns: filt = filt[filt["Entry"].isin(sel_entry)]

if filt.empty:
    st.warning("No records match the current filters.")
    st.stop()

# ── Derived subsets used across tabs ─────────────────────────────────────────
cleared_all = filt[(filt["Status"] == "Clear") &
                   filt["Waiting Days"].notna() &
                   (filt["Waiting Days"] > 0)].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Checkee.info Visa AP Statistics")
if st.session_state.get("is_sample"):
    st.warning("⚠️  Showing **synthetic sample data** — not real checkee.info records. "
               "Use **Test Connection** → **Load / Refresh Data** when the server is reachable.")
    st.caption(f"Showing **{len(filt):,}** synthetic records")
else:
    st.caption(f"Showing **{len(filt):,}** records · {start_str} → {end_str}")

total   = len(filt)
pending = int((filt["Status"] == "Pending").sum())
clear   = int((filt["Status"] == "Clear").sum())
reject  = int((filt["Status"] == "Reject").sum())
cr_pct  = f"{100*clear/(clear+reject):.1f}%" if (clear+reject) > 0 else "N/A"
med_wait = f"{cleared_all['Waiting Days'].median():.0f} d" if not cleared_all.empty else "N/A"

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Total",        f"{total:,}")
m2.metric("Pending",      f"{pending:,}")
m3.metric("Cleared",      f"{clear:,}")
m4.metric("Rejected",     f"{reject:,}")
m5.metric("Clear Rate",   cr_pct)
m6.metric("Median Wait",  med_wait)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview",
    "📍 Location Comparison",
    "🎓 Major & Degree",
    "⏱ Waiting Time Deep Dive",
    "🗂 Raw Data",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Status Distribution")
        sc = filt["Status"].value_counts().reset_index()
        sc.columns = ["Status", "Count"]
        fig = px.pie(sc, values="Count", names="Status",
                     color="Status", color_discrete_map=STATUS_COLORS, hole=0.42)
        fig.update_traces(textposition="inside", textinfo="percent+label+value",
                          pull=[0.03] * len(sc))
        fig.update_layout(margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Cases by Visa Type")
        vt = filt.groupby(["Visa Type", "Status"]).size().reset_index(name="Count")
        fig = px.bar(vt, x="Visa Type", y="Count",
                     color="Status", color_discrete_map=STATUS_COLORS,
                     barmode="stack", text_auto=True,
                     category_orders={"Visa Type": filt["Visa Type"].value_counts().index.tolist()})
        fig.update_layout(margin=dict(t=20, b=10), xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Cases by Consulate")
        cc = filt.groupby(["Consulate", "Status"]).size().reset_index(name="Count")
        fig = px.bar(cc, x="Count", y="Consulate",
                     color="Status", color_discrete_map=STATUS_COLORS,
                     barmode="stack", orientation="h", text_auto=True,
                     category_orders={"Consulate": filt["Consulate"].value_counts().index.tolist()})
        fig.update_layout(margin=dict(t=20, b=10),
                          yaxis={"categoryorder": "total ascending"}, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("New vs. Renewal")
        if "Entry" in filt.columns:
            ec = filt.groupby(["Entry", "Status"]).size().reset_index(name="Count")
            fig = px.bar(ec, x="Entry", y="Count",
                         color="Status", color_discrete_map=STATUS_COLORS,
                         barmode="group", text_auto=True)
            fig.update_layout(margin=dict(t=20, b=10), xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 25 Majors")
    if "Major" in filt.columns:
        top25 = filt["Major"].value_counts().nlargest(25).index
        md = filt[filt["Major"].isin(top25)]
        mc = md.groupby(["Major", "Status"]).size().reset_index(name="Count")
        fig = px.bar(mc, x="Count", y="Major",
                     color="Status", color_discrete_map=STATUS_COLORS,
                     barmode="stack", orientation="h", text_auto=True,
                     category_orders={"Major": md["Major"].value_counts().index.tolist()},
                     height=600)
        fig.update_layout(margin=dict(t=20, b=10),
                          yaxis={"categoryorder": "total ascending"}, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Case Trend")
    if "Month" in filt.columns:
        mon = filt.groupby(["Month", "Status"]).size().reset_index(name="Count")
        fig = px.line(mon, x="Month", y="Count",
                      color="Status", color_discrete_map=STATUS_COLORS, markers=True)
        fig.update_layout(xaxis_tickangle=-45, margin=dict(t=20, b=60))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LOCATION COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Compare consulates to decide where to apply")
    st.caption("All charts below use **cleared cases only** for wait-time analysis.")

    # ── Local filters ──────────────────────────────────────────────────────────
    lf1, lf2, lf3 = st.columns(3)
    with lf1:
        t2_visa = st.multiselect("Visa Type", visa_opts, default=visa_opts, key="t2v")
    with lf2:
        major_kw = st.text_input("Major keyword filter (e.g. CS, Biology)",
                                 placeholder="Leave blank for all", key="t2m")
    with lf3:
        t2_degree = st.multiselect("Degree level", DEGREE_ORDER,
                                   default=DEGREE_ORDER, key="t2d") if "Degree" in filt.columns else DEGREE_ORDER

    # Apply local filters
    loc_df = filt.copy()
    if t2_visa:
        loc_df = loc_df[loc_df["Visa Type"].isin(t2_visa)]
    if major_kw.strip() and "Major" in loc_df.columns:
        loc_df = loc_df[loc_df["Major"].str.contains(major_kw.strip(), case=False, na=False)]
    if "Degree" in loc_df.columns and t2_degree:
        loc_df = loc_df[loc_df["Degree"].isin(t2_degree)]

    loc_cleared = loc_df[(loc_df["Status"] == "Clear") &
                         loc_df["Waiting Days"].notna() &
                         (loc_df["Waiting Days"] > 0)]

    if loc_df.empty:
        st.warning("No data matches these filters.")
        st.stop()

    # ── Summary stats table ────────────────────────────────────────────────────
    st.subheader("Consulate Summary Table")
    st.caption("Sorted by total cases. Waiting-day stats based on cleared cases.")
    stats_tbl = location_stats(loc_df)
    st.dataframe(
        stats_tbl.style.background_gradient(subset=["Clear Rate (%)"], cmap="RdYlGn")
                       .background_gradient(subset=["Median Wait (days)"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # ── Clear-rate bar ─────────────────────────────────────────────────────────
    st.subheader("Clear Rate by Consulate")
    decided = loc_df[loc_df["Status"].isin(["Clear", "Reject"])]
    if not decided.empty:
        cr_df = (decided.groupby("Consulate")["Status"]
                 .apply(lambda x: round(100 * (x == "Clear").sum() / len(x), 1))
                 .reset_index(name="Clear Rate (%)"))
        cr_df = cr_df.sort_values("Clear Rate (%)")
        fig = px.bar(cr_df, x="Clear Rate (%)", y="Consulate", orientation="h",
                     text="Clear Rate (%)",
                     color="Clear Rate (%)",
                     color_continuous_scale=["#F44336", "#FFA500", "#4CAF50"],
                     range_color=[0, 100])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(margin=dict(t=20, b=10, r=60),
                          coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Violin plot: waiting days by consulate ─────────────────────────────────
    st.subheader("Waiting Days Distribution by Consulate (Cleared Cases)")
    if not loc_cleared.empty:
        cons_order_wait = (loc_cleared.groupby("Consulate")["Waiting Days"]
                           .median().sort_values().index.tolist())
        fig = px.violin(
            loc_cleared, x="Consulate", y="Waiting Days",
            color="Consulate", box=True, points="outliers",
            category_orders={"Consulate": cons_order_wait},
            labels={"Waiting Days": "Days to Clear"},
        )
        fig.update_layout(margin=dict(t=20, b=10), showlegend=False,
                          xaxis_title="", yaxis_title="Days to Clear")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cleared cases with waiting-day data for this filter.")

    st.divider()

    # ── ECDF: % cleared within N days ─────────────────────────────────────────
    st.subheader("Cumulative % Cleared within N Days — by Consulate")
    st.caption("Read as: *'X% of cleared cases at this consulate resolved within N days'*")
    if not loc_cleared.empty:
        consulates_with_data = loc_cleared["Consulate"].value_counts()
        consulates_with_data = consulates_with_data[consulates_with_data >= 3].index.tolist()
        palette = px.colors.qualitative.Safe
        fig = go.Figure()
        for i, cons in enumerate(consulates_with_data):
            sub = loc_cleared[loc_cleared["Consulate"] == cons]["Waiting Days"].dropna()
            if len(sub) < 3:
                continue
            trace = ecdf_traces(sub, name=cons, color=palette[i % len(palette)])
            fig.add_trace(trace)
        fig.update_layout(
            xaxis_title="Days to Clear",
            yaxis_title="Cumulative % of Cleared Cases",
            yaxis=dict(ticksuffix="%", range=[0, 102]),
            legend_title="Consulate",
            margin=dict(t=30, b=10),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough cleared cases to plot ECDF.")

    st.divider()

    # ── Heatmaps ───────────────────────────────────────────────────────────────
    h1, h2 = st.columns(2)

    with h1:
        st.subheader("Heatmap: Median Wait Days\n(Consulate × Visa Type)")
        if not loc_cleared.empty and "Visa Type" in loc_cleared.columns:
            piv = (loc_cleared.groupby(["Consulate", "Visa Type"])["Waiting Days"]
                   .median().reset_index()
                   .pivot(index="Consulate", columns="Visa Type", values="Waiting Days"))
            fig = px.imshow(piv, text_auto=".0f",
                            color_continuous_scale="RdYlGn_r",
                            labels=dict(color="Median Days"),
                            aspect="auto")
            fig.update_layout(margin=dict(t=30, b=10),
                              xaxis_title="Visa Type", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    with h2:
        st.subheader("Heatmap: Clear Rate %\n(Consulate × Visa Type)")
        decided2 = loc_df[loc_df["Status"].isin(["Clear", "Reject"])]
        if not decided2.empty and "Visa Type" in decided2.columns:
            piv2 = (decided2.groupby(["Consulate", "Visa Type"])["Status"]
                    .apply(lambda x: round(100 * (x == "Clear").sum() / len(x), 1))
                    .reset_index(name="Clear Rate")
                    .pivot(index="Consulate", columns="Visa Type", values="Clear Rate"))
            fig = px.imshow(piv2, text_auto=".0f",
                            color_continuous_scale="RdYlGn",
                            range_color=[0, 100],
                            labels=dict(color="Clear %"),
                            aspect="auto")
            fig.update_layout(margin=dict(t=30, b=10),
                              xaxis_title="Visa Type", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MAJOR & DEGREE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Analyze waiting time by field of study and degree level")
    st.caption(
        "**Degree** is inferred from the free-text Major field — look for "
        "PhD / MS / Master / BS / Bachelor keywords."
    )

    if "Degree" not in filt.columns:
        st.warning("No Major column found in data.")
        st.stop()

    # ── Local filters ──────────────────────────────────────────────────────────
    d1, d2, d3 = st.columns(3)
    with d1:
        t3_visa = st.multiselect("Visa Type", visa_opts, default=visa_opts, key="t3v")
    with d2:
        t3_kw   = st.text_input("Major keyword (e.g. Computer, Bio, EE)",
                                placeholder="Leave blank = all", key="t3kw")
    with d3:
        t3_cons = st.multiselect("Consulate", cons_opts, default=cons_opts, key="t3c")

    deg_df = filt.copy()
    if t3_visa:
        deg_df = deg_df[deg_df["Visa Type"].isin(t3_visa)]
    if t3_kw.strip():
        deg_df = deg_df[deg_df["Major"].str.contains(t3_kw.strip(), case=False, na=False)]
    if t3_cons:
        deg_df = deg_df[deg_df["Consulate"].isin(t3_cons)]

    deg_cleared = deg_df[(deg_df["Status"] == "Clear") &
                         deg_df["Waiting Days"].notna() &
                         (deg_df["Waiting Days"] > 0)]

    if deg_df.empty:
        st.warning("No data for these filters.")
        st.stop()

    # ── Degree breakdown ───────────────────────────────────────────────────────
    d_row1, d_row2 = st.columns(2)

    with d_row1:
        st.subheader("Cases by Degree Level")
        dc = deg_df.groupby(["Degree", "Status"]).size().reset_index(name="Count")
        fig = px.bar(dc, x="Degree", y="Count",
                     color="Status", color_discrete_map=STATUS_COLORS,
                     barmode="stack", text_auto=True,
                     category_orders={"Degree": DEGREE_ORDER})
        fig.update_layout(margin=dict(t=20, b=10), xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with d_row2:
        st.subheader("Clear Rate by Degree Level")
        dec_decided = deg_df[deg_df["Status"].isin(["Clear", "Reject"])]
        if not dec_decided.empty:
            dcr = (dec_decided.groupby("Degree")["Status"]
                   .apply(lambda x: round(100 * (x == "Clear").sum() / len(x), 1))
                   .reset_index(name="Clear Rate (%)"))
            dcr["Degree"] = pd.Categorical(dcr["Degree"], categories=DEGREE_ORDER, ordered=True)
            dcr = dcr.sort_values("Degree")
            fig = px.bar(dcr, x="Degree", y="Clear Rate (%)",
                         color="Degree", color_discrete_map=DEGREE_COLORS,
                         text="Clear Rate (%)",
                         category_orders={"Degree": DEGREE_ORDER})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(margin=dict(t=20, b=10), showlegend=False,
                               xaxis_title="", yaxis_range=[0, 110])
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Violin: waiting days by consulate, split by degree ────────────────────
    st.subheader("Waiting Days by Consulate — split by Degree Level")
    st.caption("Helps you see which consulate is fastest for your specific degree.")
    if not deg_cleared.empty:
        present_degrees = [d for d in DEGREE_ORDER if d in deg_cleared["Degree"].unique()]
        sel_deg_violin = st.multiselect("Degree levels to show", present_degrees,
                                        default=present_degrees, key="dv")
        vdf = deg_cleared[deg_cleared["Degree"].isin(sel_deg_violin)]
        if not vdf.empty:
            cons_ord = (vdf.groupby("Consulate")["Waiting Days"]
                        .median().sort_values().index.tolist())
            fig = px.violin(
                vdf, x="Consulate", y="Waiting Days",
                color="Degree", box=True, points=False,
                color_discrete_map=DEGREE_COLORS,
                category_orders={"Consulate": cons_ord, "Degree": DEGREE_ORDER},
                labels={"Waiting Days": "Days to Clear"},
                violinmode="overlay",
            )
            fig.update_layout(margin=dict(t=20, b=10),
                               xaxis_title="", yaxis_title="Days to Clear",
                               legend_title="Degree")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cleared cases with waiting-day data for this filter.")

    st.divider()

    # ── ECDF by degree ─────────────────────────────────────────────────────────
    st.subheader("Cumulative % Cleared within N Days — by Degree Level")
    if not deg_cleared.empty:
        fig = go.Figure()
        for deg in DEGREE_ORDER:
            sub = deg_cleared[deg_cleared["Degree"] == deg]["Waiting Days"].dropna()
            if len(sub) < 3:
                continue
            trace = ecdf_traces(sub, name=deg, color=DEGREE_COLORS[deg])
            fig.add_trace(trace)
        fig.update_layout(
            xaxis_title="Days to Clear",
            yaxis_title="Cumulative %",
            yaxis=dict(ticksuffix="%", range=[0, 102]),
            legend_title="Degree",
            hovermode="x unified",
            margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Top majors by degree ───────────────────────────────────────────────────
    st.subheader("Top Majors within Selected Degree Levels")
    if not deg_df.empty:
        top_n = st.slider("Show top N majors", 10, 40, 20, key="topn")
        top_maj = deg_df["Major"].value_counts().nlargest(top_n).index
        mdf = deg_df[deg_df["Major"].isin(top_maj)]
        mc  = mdf.groupby(["Major", "Degree"]).size().reset_index(name="Count")
        maj_ord = mdf["Major"].value_counts().index.tolist()
        fig = px.bar(mc, x="Count", y="Major",
                     color="Degree", color_discrete_map=DEGREE_COLORS,
                     barmode="stack", orientation="h",
                     category_orders={"Major": maj_ord, "Degree": DEGREE_ORDER},
                     height=max(400, top_n * 22))
        fig.update_layout(margin=dict(t=20, b=10),
                          yaxis={"categoryorder": "total ascending"}, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — WAITING TIME DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Detailed waiting time analysis (cleared cases only)")

    if cleared_all.empty:
        st.info("No cleared cases with waiting-day data in current filter.")
        st.stop()

    # ── Percentile table ───────────────────────────────────────────────────────
    st.subheader("Percentile Table by Consulate")
    pct_rows = []
    for cons, grp in cleared_all.groupby("Consulate"):
        wd = grp["Waiting Days"].dropna()
        if len(wd) < 2:
            continue
        pct_rows.append({
            "Consulate": cons, "N":  len(wd),
            "Min": int(wd.min()), "25th": int(wd.quantile(0.25)),
            "Median": int(wd.median()),
            "75th": int(wd.quantile(0.75)),
            "90th": int(wd.quantile(0.90)),
            "Max": int(wd.max()),
            "Mean": round(wd.mean(), 1),
        })
    if pct_rows:
        pct_df = pd.DataFrame(pct_rows).sort_values("Median")
        st.dataframe(
            pct_df.style.background_gradient(subset=["Median", "75th", "90th"],
                                             cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    # ── Histogram with marginal box ────────────────────────────────────────────
    st.subheader("Overall Waiting-Days Histogram (Cleared Cases)")
    color_by = st.radio("Color by", ["Visa Type", "Consulate", "Degree"],
                        horizontal=True, key="hist_col")
    if color_by not in cleared_all.columns:
        color_by = "Visa Type"
    fig = px.histogram(
        cleared_all, x="Waiting Days", color=color_by,
        nbins=60, opacity=0.72, marginal="box",
        labels={"Waiting Days": "Days to Clear"},
        barmode="overlay",
    )
    fig.update_layout(margin=dict(t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Box: waiting days by visa type ────────────────────────────────────────
    st.subheader("Box Plot by Visa Type")
    fig = px.box(cleared_all, x="Visa Type", y="Waiting Days",
                 color="Visa Type", points="outliers")
    fig.update_layout(margin=dict(t=20, b=10), showlegend=False, xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Scatter: check date vs waiting days ────────────────────────────────────
    st.subheader("Waiting Days vs. Check Date (Cleared Cases)")
    st.caption("Reveals whether processing speed is improving or worsening over time.")
    if "Check Date" in cleared_all.columns:
        sc_color = st.radio("Color by", ["Consulate", "Visa Type"], horizontal=True, key="sc_col")
        sc_df = cleared_all[cleared_all["Check Date"].notna()].copy()
        if not sc_df.empty:
            fig = px.scatter(
                sc_df, x="Check Date", y="Waiting Days",
                color=sc_color, opacity=0.6,
                labels={"Waiting Days": "Days to Clear"},
                trendline="lowess", trendline_scope="overall",
                trendline_color_override="black",
            )
            fig.update_layout(margin=dict(t=20, b=10), legend_title=sc_color)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Monthly median wait trend ──────────────────────────────────────────────
    st.subheader("Monthly Median Waiting Days by Consulate")
    if "Month" in cleared_all.columns:
        mw = (cleared_all.groupby(["Month", "Consulate"])["Waiting Days"]
              .median().reset_index(name="Median Wait Days"))
        fig = px.line(mw, x="Month", y="Median Wait Days", color="Consulate",
                      markers=True)
        fig.update_layout(xaxis_tickangle=-45, margin=dict(t=20, b=60))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Raw Data Table")
    st.dataframe(filt.reset_index(drop=True), use_container_width=True, height=500)
    csv = filt.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Download CSV", csv,
                       file_name="checkee_data.csv", mime="text/csv")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Urban Air Quality × Respiratory Health — Enhanced EDA Streamlit Dashboard  ║
# ║  Dataset : India CPCB city_day.csv (2015–2020)                              ║
# ║  Run     : streamlit run streamlit_app.py                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import warnings, io
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import streamlit as st
import plotly.express     as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import requests

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirHealth EDA India",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  CUSTOM CSS  – refined dark editorial theme
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;600;700;800&family=Playfair+Display:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* ── Core Background ── */
.stApp {
    background-color: #080b12;
    background-image:
        radial-gradient(ellipse 80% 50% at 10% 0%, rgba(0,183,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 100%, rgba(255,59,92,0.05) 0%, transparent 60%);
    color: #dde3f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #06080f;
    border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #8892a4;
    font-size: 0.78rem;
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0e1321 0%, #111827 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00b7ff, #00ffd5);
    opacity: 0.7;
}
[data-testid="metric-container"] label {
    color: #4a5568 !important;
    font-size: .68rem !important;
    letter-spacing: .12em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8edf8 !important;
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: .75rem !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Typography ── */
h1 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 800 !important;
    color: #f0f4ff !important;
    letter-spacing: -.01em;
    line-height: 1.15 !important;
}
h2 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    color: #dde3f0 !important;
    letter-spacing: -.01em;
}
h3 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    color: #c5cee0 !important;
}

/* ── Sidebar Labels ── */
.stSelectbox label, .stMultiSelect label, .stSlider label,
.stDateInput label, .stRadio label {
    color: #4a5568 !important;
    font-size: .7rem !important;
    letter-spacing: .1em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #0e1321 !important;
    border-color: rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    gap: 0;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #4a5568;
    padding: .65rem 1.25rem;
    font-family: 'DM Mono', monospace;
    font-size: .73rem;
    font-weight: 500;
    letter-spacing: .06em;
    text-transform: uppercase;
    border-bottom: 2px solid transparent;
    transition: color .2s, border-color .2s;
}
.stTabs [aria-selected="true"] {
    color: #00b7ff !important;
    border-bottom-color: #00b7ff !important;
    background: transparent !important;
}

/* ── Alerts / Info Boxes ── */
.stAlert {
    background: #0e1321 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
}

/* ── Dividers ── */
hr { border-color: rgba(255,255,255,0.05) !important; }

/* ── DataFrames ── */
.stDataFrame {
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00b7ff !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080b12; }
::-webkit-scrollbar-thumb { background: #1e2a3a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2a3a50; }

/* ── Stat Badge ── */
.stat-badge {
    background: linear-gradient(135deg, #0e1321, #111827);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.stat-badge::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,183,255,0.3), transparent);
}

/* ── Section Heading ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: .65rem;
    letter-spacing: .18em;
    color: #2d3a4f;
    text-transform: uppercase;
    margin-bottom: .3rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  PLOTLY DEFAULTS
# ─────────────────────────────────────────────────────────────────
COLORS = [
    "#00b7ff", "#ff3b5c", "#00ffd5", "#ffb800",
    "#a855f7", "#ff6b35", "#22d3ee", "#f472b6",
    "#84cc16", "#fb923c"
]

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Outfit, sans-serif", color="#8892a4", size=12),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.06)",
        tickfont=dict(size=11),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.06)",
        tickfont=dict(size=11),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.06)",
        font=dict(size=11),
    ),
    colorway=COLORS,
    margin=dict(t=48, b=44, l=54, r=24),
    hoverlabel=dict(
        bgcolor="#0e1321",
        bordercolor="rgba(255,255,255,0.1)",
        font=dict(family="DM Mono, monospace", size=11, color="#dde3f0"),
    ),
)

def layout(**kw):
    base = {**LAYOUT}
    if "xaxis" in kw:
        base["xaxis"] = {**LAYOUT["xaxis"], **kw.pop("xaxis")}
    if "yaxis" in kw:
        base["yaxis"] = {**LAYOUT["yaxis"], **kw.pop("yaxis")}
    return {**base, **kw}


# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def _aqi_color(val):
    """Return a color for a given AQI value."""
    if pd.isna(val):   return "#4a5568"
    if val <= 50:      return "#22c55e"
    if val <= 100:     return "#84cc16"
    if val <= 200:     return "#facc15"
    if val <= 300:     return "#f97316"
    if val <= 400:     return "#ef4444"
    return "#c026d3"

def _pollutant_unit(pol):
    return "ppm" if pol == "CO" else "µg/m³"


# ─────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────
DATA_URLS = [
    "https://raw.githubusercontent.com/AnirudhBHarish/India-Air-Quality-Analysis/main/city_day.csv",
    "https://raw.githubusercontent.com/ChiaPatricia/Predicting-Air-Quality-Index-in-India/master/city_day.csv",
    "https://raw.githubusercontent.com/eeshwarib23/DataViz-IndiaAirQuality-Python-D3.js-HTML/main/data/city_day.csv",
    "https://raw.githubusercontent.com/pranitagg/air-quality-dataSet/master/city_day.csv",
]

THRESHOLDS = {
    "PM2.5": {"WHO": 15,  "NAAQS": 60},
    "PM10" : {"WHO": 45,  "NAAQS": 100},
    "NO2"  : {"WHO": 25,  "NAAQS": 80},
    "SO2"  : {"WHO": 40,  "NAAQS": 80},
    "CO"   : {"WHO": 4,   "NAAQS": 2},
    "O3"   : {"WHO": 100, "NAAQS": 100},
}

AQI_BUCKETS = {
    "Good"        : (0,   50,  "#22c55e"),
    "Satisfactory": (51,  100, "#84cc16"),
    "Moderate"    : (101, 200, "#facc15"),
    "Poor"        : (201, 300, "#f97316"),
    "Very Poor"   : (301, 400, "#ef4444"),
    "Severe"      : (401, 999, "#c026d3"),
}

SEASON_COLORS = {
    "Winter": "#00b7ff",
    "Spring": "#22d3ee",
    "Summer": "#ffb800",
    "Autumn": "#ff6b35",
}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = None
    for url in DATA_URLS:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [c.strip() for c in df.columns]
                if "City" in df.columns and "Date" in df.columns:
                    break
                else:
                    df = None
        except Exception:
            continue

    if df is None:
        df = _generate_fallback()

    return _clean(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "Datetime": "Date", "datetime": "Date", "date": "Date",
        "pm2_5": "PM2.5", "pm25": "PM2.5", "pm10": "PM10",
        "no2": "NO2", "so2": "SO2", "co": "CO", "o3": "O3",
        "aqi": "AQI", "AQI_Bucket": "AQI_Bucket", "aqi_bucket": "AQI_Bucket",
        "city": "City", "location": "City", "state": "State",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    for col in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]:
        if col not in df.columns:
            df[col] = np.nan

    num_cols = ["PM2.5","PM10","NO2","SO2","CO","O3","AQI",
                "NO","NOx","NH3","Benzene","Toluene","Xylene"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] < 0, c] = np.nan

    for c in ["PM2.5","PM10","NO2","SO2","CO","O3","AQI"]:
        if c in df.columns:
            cap = df[c].quantile(0.995)
            df.loc[df[c] > cap, c] = cap

    if "City" in df.columns:
        parts = []
        for city, grp in df.groupby("City", group_keys=False):
            grp = grp.copy().sort_values("Date").reset_index(drop=True)
            num = grp.select_dtypes(include=[np.number]).columns.tolist()
            grp[num] = (
                grp.set_index("Date")[num]
                   .interpolate(method="time", limit=7)
                   .reset_index(drop=True)
            )
            parts.append(grp)
        df = pd.concat(parts, ignore_index=True)
    else:
        df["City"] = "Unknown"

    if "AQI_Bucket" not in df.columns or df["AQI_Bucket"].isna().all():
        def bucket(v):
            if pd.isna(v): return np.nan
            for name, (lo, hi, _) in AQI_BUCKETS.items():
                if lo <= v <= hi: return name
            return "Severe"
        df["AQI_Bucket"] = df["AQI"].apply(bucket)

    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["DayOfWeek"] = df["Date"].dt.day_name()
    df["Quarter"]   = df["Date"].dt.quarter
    df["Season"]    = df["Month"].map({
        12:"Winter", 1:"Winter", 2:"Winter",
        3:"Spring",  4:"Spring", 5:"Spring",
        6:"Summer",  7:"Summer", 8:"Summer",
        9:"Autumn", 10:"Autumn",11:"Autumn",
    })

    _add_health_proxy(df)
    return df


def _add_health_proxy(df: pd.DataFrame) -> None:
    CITY_POP = {
        "Delhi": 30e6, "Ahmedabad": 8e6, "Mumbai": 21e6, "Kolkata": 15e6,
        "Chennai": 11e6, "Bengaluru": 13e6, "Hyderabad": 10e6, "Lucknow": 4e6,
        "Patna": 3e6, "Jaipur": 4e6, "Chandigarh": 1e6, "Bhopal": 2e6,
    }
    if "City" not in df.columns:
        df["resp_cases"] = np.nan
        return

    cases_all = []
    for city, grp in df.groupby("City"):
        grp = grp.copy().sort_values("Date").reset_index(drop=True)
        pop       = CITY_POP.get(city, 2e6)
        base_rate = 0.0007 * (pop / 1e6)

        pm25 = grp["PM2.5"].fillna(grp["PM2.5"].median())
        no2  = grp["NO2"].fillna(grp["NO2"].median())
        pm10 = grp["PM10"].fillna(grp["PM10"].median())

        pm25_lag = pm25.shift(3).fillna(pm25.mean())
        pm10_lag = pm10.shift(3).fillna(pm10.mean())
        no2_lag  = no2.shift(3).fillna(no2.mean())

        def znorm(s):
            std = s.std()
            return (s - s.mean()) / std if std > 0 else s * 0

        sig = 0.45 * znorm(pm25_lag) + 0.30 * znorm(pm10_lag) + 0.25 * znorm(no2_lag)
        np.random.seed(abs(hash(city)) % 2**31)
        noise = np.random.normal(0, 0.1, len(grp))
        cases = np.maximum(0, base_rate * (1 + 0.55 * sig + noise)).round().astype(int)
        grp["resp_cases"] = cases
        cases_all.append(grp[["Date", "City", "resp_cases"]])

    cases_df = pd.concat(cases_all).set_index(["Date", "City"])["resp_cases"]
    df["resp_cases"] = (
        pd.MultiIndex.from_arrays([df["Date"], df["City"]])
          .map(cases_df.to_dict())
    )
    df["resp_cases"] = pd.to_numeric(df["resp_cases"], errors="coerce")


def _generate_fallback() -> pd.DataFrame:
    import itertools
    np.random.seed(42)
    cities = [
        "Delhi","Mumbai","Bengaluru","Kolkata","Chennai",
        "Ahmedabad","Lucknow","Jaipur","Patna","Hyderabad",
    ]
    dates = pd.date_range("2015-01-01", "2020-06-30", freq="D")
    base = {
        "Delhi"     : {"PM2.5":95, "PM10":175,"NO2":58,"SO2":18,"CO":1.6,"O3":35},
        "Mumbai"    : {"PM2.5":48, "PM10":88, "NO2":42,"SO2":10,"CO":1.1,"O3":40},
        "Bengaluru" : {"PM2.5":32, "PM10":62, "NO2":36,"SO2":7, "CO":0.9,"O3":38},
        "Kolkata"   : {"PM2.5":72, "PM10":138,"NO2":52,"SO2":16,"CO":1.4,"O3":30},
        "Chennai"   : {"PM2.5":38, "PM10":70, "NO2":33,"SO2":8, "CO":0.8,"O3":42},
        "Ahmedabad" : {"PM2.5":85, "PM10":160,"NO2":50,"SO2":20,"CO":1.5,"O3":32},
        "Lucknow"   : {"PM2.5":78, "PM10":148,"NO2":48,"SO2":15,"CO":1.3,"O3":28},
        "Jaipur"    : {"PM2.5":65, "PM10":120,"NO2":40,"SO2":12,"CO":1.1,"O3":33},
        "Patna"     : {"PM2.5":90, "PM10":168,"NO2":55,"SO2":18,"CO":1.5,"O3":25},
        "Hyderabad" : {"PM2.5":45, "PM10":85, "NO2":38,"SO2":9, "CO":1.0,"O3":41},
    }
    rows = []
    for city, _ in itertools.islice(base.items(), 10):
        b = base[city]
        for date in dates:
            m  = date.month
            sf = [1.8,1.6,1.2,0.9,0.7,0.6,0.5,0.6,0.8,1.1,1.5,1.9][m - 1]
            rows.append({
                "City" : city, "Date" : date,
                "PM2.5": max(1,   b["PM2.5"] * sf * np.random.lognormal(0, .25)),
                "PM10" : max(2,   b["PM10"]  * sf * np.random.lognormal(0, .22)),
                "NO2"  : max(1,   b["NO2"]   * sf * np.random.lognormal(0, .20)),
                "SO2"  : max(0.5, b["SO2"]   * sf * np.random.lognormal(0, .30)),
                "CO"   : max(0.1, b["CO"]    * sf * np.random.lognormal(0, .25)),
                "O3"   : max(5,   b["O3"] * (1.1 - 0.1 * sf) * np.random.lognormal(0, .15)),
                "AQI"  : np.nan,
            })
    df = pd.DataFrame(rows)
    df["AQI"] = (df["PM2.5"] * 2.0 + df["PM10"] * 0.5).round(1)
    return df


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    with st.spinner("Fetching India CPCB Air Quality dataset…"):
        df_full = load_data()

    all_cities     = sorted(df_full["City"].dropna().unique().tolist())
    all_pollutants = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3"] if c in df_full.columns]

    # ─────────────────────────────────────────────────────────────
    #  SIDEBAR
    # ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='padding:.4rem 0 1.2rem'>"
            "<div style='font-family:Playfair Display,serif;font-size:1.2rem;"
            "font-weight:800;color:#f0f4ff;letter-spacing:-.01em'>AirHealth EDA</div>"
            "<div style='font-family:DM Mono,monospace;font-size:.65rem;"
            "color:#2d3a4f;letter-spacing:.12em;text-transform:uppercase;"
            "margin-top:.2rem'>India · 2015–2020</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        selected_cities = st.multiselect(
            "Cities", all_cities,
            default=all_cities[:6] if len(all_cities) >= 6 else all_cities,
        )
        if not selected_cities:
            selected_cities = all_cities[:3]

        selected_pollutant = st.selectbox("Primary Pollutant", all_pollutants)

        date_min = df_full["Date"].min().date()
        date_max = df_full["Date"].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
        )
        start_date = pd.Timestamp(date_range[0])
        end_date   = pd.Timestamp(date_range[1] if len(date_range) > 1 else date_max)

        lag_days = st.slider("Lag Days (pollution → health)", 0, 14, 3)

        st.divider()
        st.markdown(
            "<div style='font-family:DM Mono,monospace;font-size:.65rem;"
            "color:#2d3a4f;line-height:1.6'>⚠ Respiratory case data is modelled "
            "using an epidemiological lag function — not real hospital records.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='margin-top:1.5rem;font-family:DM Mono,monospace;"
            "font-size:.62rem;color:#1e2a3a'>Source: CPCB via GitHub</div>",
            unsafe_allow_html=True,
        )

    # ── Filtered dataframe ──
    mask = (
        df_full["City"].isin(selected_cities) &
        (df_full["Date"] >= start_date) &
        (df_full["Date"] <= end_date)
    )
    df = df_full[mask].copy()

    if df.empty:
        st.error("No data for selected filters — adjust the sidebar.")
        return

    # ─────────────────────────────────────────────────────────────
    #  HEADER
    # ─────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='padding:2rem 0 1rem'>"
        "<h1 style='font-size:2.4rem;margin-bottom:.3rem'>Urban Air Quality<br>"
        "<span style='color:#00b7ff'>× Respiratory Health</span></h1>"
        "<p style='font-family:DM Mono,monospace;font-size:.72rem;"
        "color:#2d3a4f;letter-spacing:.1em;text-transform:uppercase;margin-top:.2rem'>"
        "India CPCB · 2015–2020 · 26 Cities · PM2.5 · PM10 · NO₂ · SO₂ · CO · O₃ · Full EDA + Lag Analysis"
        "</p></div>",
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────
    #  TABS
    # ─────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "Overview", "Pollution Trends", "Health Proxy",
        "Lag Analysis", "Correlations", "City Comparison",
        "EDA Deep Dive",
    ])

    with tabs[0]: _tab_overview(df, selected_pollutant)
    with tabs[1]: _tab_trends(df, selected_cities, selected_pollutant)
    with tabs[2]: _tab_health(df, selected_pollutant)
    with tabs[3]: _tab_lag(df, selected_pollutant, lag_days)
    with tabs[4]: _tab_correlations(df, selected_pollutant)
    with tabs[5]: _tab_cities(df, all_pollutants, selected_cities)
    with tabs[6]: _tab_eda(df, selected_pollutant)


# ─────────────────────────────────────────────────────────────────
#  TAB IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────

def _tab_overview(df, pol):
    pol_data = df[pol].dropna()
    aqi_data = df["AQI"].dropna()
    unit     = _pollutant_unit(pol)

    # ── Key metrics ──
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(f"Avg {pol}",       f"{pol_data.mean():.1f} {unit}",  f"max {pol_data.max():.0f}")
    c2.metric("Avg AQI",          f"{aqi_data.mean():.0f}",          f"max {aqi_data.max():.0f}")
    c3.metric("Cities",           str(df["City"].nunique()))
    c4.metric("Records",          f"{len(df):,}")
    c5.metric("Period",
              f"{df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}")
    c6.metric(f"Missing {pol}",   f"{df[pol].isna().sum():,}",
              f"{df[pol].isna().mean() * 100:.1f}%")

    st.divider()

    col1, col2 = st.columns([1, 1.15])

    # ── AQI donut ──
    with col1:
        bucket_counts = df["AQI_Bucket"].value_counts().reset_index()
        bucket_counts.columns = ["Bucket", "Count"]
        bcolors = {k: v[2] for k, v in AQI_BUCKETS.items()}
        fig = px.pie(
            bucket_counts, names="Bucket", values="Count",
            color="Bucket", color_discrete_map=bcolors,
            title="AQI Bucket Distribution",
            hole=0.52,
        )
        fig.update_layout(**layout(
            title_font_size=13, title_font_family="Outfit, sans-serif",
            height=340,
            annotations=[dict(
                text=f"<b>{aqi_data.mean():.0f}</b><br><span style='font-size:10px'>Avg AQI</span>",
                x=0.5, y=0.5, font_size=16, showarrow=False,
                font_color="#dde3f0",
            )],
        ))
        fig.update_traces(
            textinfo="percent",
            textfont=dict(family="DM Mono, monospace", size=10),
            marker=dict(line=dict(color="#080b12", width=2)),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Pollutant averages ──
    with col2:
        pols  = ["PM2.5","PM10","NO2","SO2","CO","O3"]
        avgs  = [df[p].mean() if p in df.columns else 0 for p in pols]
        # Normalise for colour gradient
        norm_avgs = np.array(avgs)
        norm_avgs = (norm_avgs - norm_avgs.min()) / (norm_avgs.max() - norm_avgs.min() + 1e-9)
        bar_colors = [f"rgba({int(0+r*255)},{int(183-r*183)},{int(255-r*255)},0.85)"
                      for r in norm_avgs]
        fig2 = go.Figure(go.Bar(
            x=pols, y=avgs,
            marker_color=bar_colors,
            marker_line_width=0,
            text=[f"{v:.1f}" for v in avgs],
            textposition="outside",
            textfont=dict(family="DM Mono, monospace", size=10, color="#8892a4"),
        ))
        fig2.update_layout(**layout(
            title="Average Pollutant Levels",
            title_font_size=13,
            showlegend=False,
            yaxis_title="Concentration",
            height=340,
        ))
        st.plotly_chart(fig2, use_container_width=True)

    # ── WHO / NAAQS exceedance ──
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:.68rem;"
        "letter-spacing:.12em;color:#2d3a4f;text-transform:uppercase;"
        "margin:1.2rem 0 .6rem'>WHO & NAAQS Exceedance Analysis</div>",
        unsafe_allow_html=True,
    )
    exc_pols = [p for p in ["PM2.5","PM10","NO2","SO2"] if p in df.columns]
    cols = st.columns(len(exc_pols))
    for i, p in enumerate(exc_pols):
        thr      = THRESHOLDS.get(p, {})
        val      = df[p].dropna()
        who_exc  = (val > thr.get("WHO",  9999)).mean() * 100
        naaq_exc = (val > thr.get("NAAQS", 9999)).mean() * 100
        color    = "#ef4444" if who_exc > 60 else "#f97316" if who_exc > 30 else "#22c55e"
        with cols[i]:
            st.markdown(
                f"<div class='stat-badge'>"
                f"<div style='font-family:DM Mono,monospace;font-size:.6rem;"
                f"color:#2d3a4f;letter-spacing:.12em;text-transform:uppercase'>{p}</div>"
                f"<div style='font-size:1.8rem;font-weight:800;color:{color};"
                f"font-family:Outfit,sans-serif;margin:.3rem 0 .1rem'>{who_exc:.0f}%</div>"
                f"<div style='font-size:.67rem;color:#2d3a4f;font-family:DM Mono,monospace'>"
                f"days above WHO {thr.get('WHO','?')} {_pollutant_unit(p)}</div>"
                f"<div style='font-size:.9rem;font-weight:600;color:#fb923c;"
                f"font-family:Outfit,sans-serif;margin-top:.5rem'>{naaq_exc:.0f}%</div>"
                f"<div style='font-size:.62rem;color:#2d3a4f;font-family:DM Mono,monospace'>"
                f"days above NAAQS {thr.get('NAAQS','?')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Data preview ──
    st.markdown("<div style='margin-top:1.6rem'></div>", unsafe_allow_html=True)
    with st.expander("📋 Dataset Preview", expanded=False):
        st.markdown(
            "<div style='font-family:DM Mono,monospace;font-size:.7rem;color:#4a5568;"
            "padding:.6rem 0'>"
            "India CPCB (Central Pollution Control Board) · 26 cities · Daily readings · "
            "Columns: City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, "
            "Benzene, Toluene, Xylene, AQI, AQI_Bucket"
            "</div>",
            unsafe_allow_html=True,
        )
        display_cols = [c for c in
                        ["City","Date","PM2.5","PM10","NO2","SO2","CO","O3","AQI","AQI_Bucket"]
                        if c in df.columns]
        st.dataframe(df[display_cols].head(200), use_container_width=True, height=280)


def _tab_trends(df, cities, pol):
    unit = _pollutant_unit(pol)

    # ── Daily time series ──
    st.markdown(
        f"<h3 style='font-size:1rem;margin-bottom:.8rem'>Daily {pol} — All Selected Cities</h3>",
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    for i, city in enumerate(cities):
        grp = df[df["City"] == city].sort_values("Date")
        if grp[pol].dropna().empty:
            continue
        clr = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=grp["Date"], y=grp[pol],
            mode="lines", name=city,
            line=dict(color=clr, width=1),
            opacity=0.2, legendgroup=city, showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=grp["Date"], y=grp[pol].rolling(7, min_periods=1).mean(),
            mode="lines", name=f"{city}",
            line=dict(color=clr, width=2.2),
            legendgroup=city,
            hovertemplate=f"<b>{city}</b><br>Date: %{{x|%Y-%m-%d}}<br>{pol}: %{{y:.1f}} {unit}<extra></extra>",
        ))
    fig.update_layout(**layout(
        title=f"{pol} — 7-day Rolling Average",
        xaxis_title="", yaxis_title=f"{pol} ({unit})",
        height=420,
    ))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    # ── Monthly seasonal bar ──
    with col1:
        monthly = (df.groupby(["MonthName","Month"])[pol]
                     .mean().reset_index()
                     .sort_values("Month"))
        fig2 = px.bar(
            monthly, x="MonthName", y=pol,
            color=pol, color_continuous_scale=[
                [0, "#22c55e"], [0.4, "#facc15"],
                [0.7, "#f97316"], [1, "#ef4444"],
            ],
            title=f"Seasonal Pattern — Avg Monthly {pol}",
            labels={pol: f"{pol} ({unit})"},
        )
        fig2.update_layout(**layout(
            height=330, title_font_size=13,
            coloraxis_showscale=False,
        ))
        fig2.update_traces(
            text=monthly[pol].round(1), textposition="outside",
            textfont=dict(family="DM Mono, monospace", size=9),
            marker_line_width=0,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Year-over-year ──
    with col2:
        yearly = df.groupby("Year")[pol].mean().reset_index()
        fig3   = go.Figure(go.Scatter(
            x=yearly["Year"], y=yearly[pol],
            mode="lines+markers+text",
            text=yearly[pol].round(1),
            textposition="top center",
            textfont=dict(family="DM Mono, monospace", size=10, color="#8892a4"),
            line=dict(color="#00b7ff", width=2.5),
            marker=dict(size=9, color="#ff3b5c",
                        line=dict(color="#080b12", width=2)),
            fill="tozeroy",
            fillcolor="rgba(0,183,255,0.05)",
        ))
        fig3.update_layout(**layout(
            title=f"Year-over-Year {pol} Trend",
            height=330, title_font_size=13,
            xaxis_title="Year", yaxis_title=f"{pol} ({unit})",
            xaxis=dict(dtick=1),
        ))
        st.plotly_chart(fig3, use_container_width=True)

    # ── City × Month heatmap ──
    st.markdown(
        f"<h3 style='font-size:1rem;margin:.8rem 0'>Monthly Heatmap — City × Month</h3>",
        unsafe_allow_html=True,
    )
    month_names = {i: name for i, name in enumerate(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 1)}
    pivot = (df.groupby(["City","Month"])[pol]
               .mean().reset_index()
               .pivot(index="City", columns="Month", values=pol)
               .rename(columns=month_names))
    fig4 = px.imshow(
        pivot,
        color_continuous_scale=[
            [0.0, "#0a2540"], [0.3, "#0f4c75"],
            [0.6, "#f97316"], [1.0, "#ef4444"],
        ],
        title=f"City × Month — Avg {pol} ({unit})",
        labels=dict(color=pol),
        aspect="auto",
        text_auto=".0f",
    )
    fig4.update_layout(**layout(
        height=360, title_font_size=13,
        coloraxis_colorbar=dict(thickness=12, len=0.8),
    ))
    fig4.update_traces(textfont=dict(size=9, family="DM Mono, monospace"))
    st.plotly_chart(fig4, use_container_width=True)

    # ── Rolling stats ──
    st.markdown(
        "<h3 style='font-size:1rem;margin:.8rem 0'>Rolling Statistics</h3>",
        unsafe_allow_html=True,
    )
    city_sel = st.selectbox("City", sorted(df["City"].unique()), key="roll_city")
    grp = df[df["City"] == city_sel].sort_values("Date")
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=grp["Date"], y=grp[pol],
        mode="lines", name="Daily",
        line=dict(color="rgba(0,183,255,0.2)", width=1),
        fill="tozeroy", fillcolor="rgba(0,183,255,0.03)",
    ))
    for window, color, dash in [(7,"#00b7ff","solid"),(14,"#ffb800","dot"),(30,"#ff3b5c","dash")]:
        fig5.add_trace(go.Scatter(
            x=grp["Date"], y=grp[pol].rolling(window, min_periods=1).mean(),
            mode="lines", name=f"{window}d MA",
            line=dict(color=color, width=2.2, dash=dash),
        ))
    fig5.update_layout(**layout(
        title=f"{city_sel} — {pol} Rolling Averages (7d / 14d / 30d)",
        xaxis_title="", yaxis_title=f"{pol} ({unit})", height=380,
    ))
    st.plotly_chart(fig5, use_container_width=True)


def _tab_health(df, pol):
    unit = _pollutant_unit(pol)

    st.markdown(
        "<div style='background:#0e1321;border:1px solid rgba(0,183,255,0.12);"
        "border-left:3px solid #00b7ff;border-radius:8px;padding:.8rem 1rem;"
        "font-family:DM Mono,monospace;font-size:.72rem;color:#4a5568;"
        "line-height:1.7;margin-bottom:1.2rem'>"
        "⚕ Respiratory case counts are modelled via an epidemiological lag function "
        "(PM2.5, PM10, NO₂ at t−3 → cases at t), mirroring patterns from peer-reviewed "
        "literature. Real hospital admission data is not publicly available at this granularity."
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        city_sel = st.selectbox("City", sorted(df["City"].unique()), key="health_city")
        grp = df[df["City"] == city_sel].sort_values("Date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=grp["Date"], y=grp["resp_cases"],
            mode="lines", name="Daily",
            line=dict(color="rgba(255,59,92,0.25)", width=1),
            fill="tozeroy", fillcolor="rgba(255,59,92,0.04)",
        ))
        fig.add_trace(go.Scatter(
            x=grp["Date"], y=grp["resp_cases"].rolling(7, min_periods=1).mean(),
            mode="lines", name="7d MA", line=dict(color="#ff3b5c", width=2.5),
        ))
        fig.update_layout(**layout(
            title=f"{city_sel} — Respiratory Cases (Modelled)",
            xaxis_title="", yaxis_title="Cases / day", height=320,
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(
            x=grp["Date"],
            y=grp[pol].rolling(7, min_periods=1).mean(),
            name=f"{pol} (7d MA)",
            line=dict(color="#00b7ff", width=2.5),
            hovertemplate=f"{pol}: %{{y:.1f}} {unit}<extra></extra>",
        ), secondary_y=False)
        fig2.add_trace(go.Scatter(
            x=grp["Date"],
            y=grp["resp_cases"].rolling(7, min_periods=1).mean(),
            name="Cases (7d MA)",
            line=dict(color="#ff3b5c", width=2.5),
            hovertemplate="Cases: %{y:.0f}<extra></extra>",
        ), secondary_y=True)
        fig2.update_layout(**layout(
            title=f"{city_sel} — {pol} vs Cases (7d MA)", height=320,
        ))
        fig2.update_yaxes(title_text=f"{pol} ({unit})", secondary_y=False,
                          gridcolor="rgba(255,255,255,0.04)", color="#00b7ff")
        fig2.update_yaxes(title_text="Cases / day", secondary_y=True,
                          gridcolor="rgba(0,0,0,0)", color="#ff3b5c")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Seasonal health burden ──
    st.markdown("<h3 style='font-size:1rem;margin:.8rem 0'>Seasonal Health Burden</h3>",
                unsafe_allow_html=True)
    monthly_h = (df.groupby(["MonthName","Month","Season"])["resp_cases"]
                   .mean().reset_index().sort_values("Month"))
    fig3 = px.bar(
        monthly_h, x="MonthName", y="resp_cases",
        color="Season", color_discrete_map=SEASON_COLORS,
        title="Avg Daily Respiratory Cases by Month",
        labels={"resp_cases": "Cases / day"},
    )
    fig3.update_layout(**layout(height=320, title_font_size=13))
    fig3.update_traces(marker_line_width=0)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Scatter ──
    st.markdown(f"<h3 style='font-size:1rem;margin:.8rem 0'>{pol} vs Respiratory Cases</h3>",
                unsafe_allow_html=True)
    sample = df[[pol,"resp_cases","City","AQI_Bucket"]].dropna().sample(
        min(3000, len(df)), random_state=1,
    )
    fig4 = go.Figure()
    for i, city in enumerate(sample["City"].unique()):
        grp_s = sample[sample["City"] == city]
        fig4.add_trace(go.Scatter(
            x=grp_s[pol], y=grp_s["resp_cases"],
            mode="markers", name=city,
            marker=dict(color=COLORS[i % len(COLORS)], size=4.5,
                        opacity=0.5, line=dict(width=0)),
        ))
    _x = sample[pol].values
    _y = sample["resp_cases"].values
    _slope, _intercept, _r, _p, _ = stats.linregress(_x, _y)
    _xline = np.linspace(_x.min(), _x.max(), 200)
    fig4.add_trace(go.Scatter(
        x=_xline, y=_slope * _xline + _intercept,
        mode="lines", name=f"Trend (r={_r:.3f})",
        line=dict(color="rgba(255,255,255,0.6)", width=2, dash="dash"),
    ))
    fig4.update_layout(**layout(
        title=f"{pol} vs Respiratory Cases (Pearson r = {_r:.3f})",
        xaxis_title=f"{pol} ({unit})", yaxis_title="Cases / day",
        height=400, title_font_size=13,
    ))
    st.plotly_chart(fig4, use_container_width=True)


def _tab_lag(df, pol, lag_days):
    unit = _pollutant_unit(pol)

    st.markdown(
        f"<div style='font-family:DM Mono,monospace;font-size:.75rem;"
        f"color:#4a5568;line-height:1.8;margin-bottom:1rem'>"
        f"Pollution at day <b style='color:#00b7ff'>t</b> → "
        f"Respiratory cases at day <b style='color:#ff3b5c'>t + n</b> · "
        f"Primary predictor: <b style='color:#dde3f0'>{pol}</b> · "
        f"Selected lag: <b style='color:#dde3f0'>{lag_days}d</b></div>",
        unsafe_allow_html=True,
    )

    city_sel = st.selectbox("City", sorted(df["City"].unique()), key="lag_city")
    grp = df[df["City"] == city_sel].sort_values("Date").reset_index(drop=True)

    max_lag = 14
    results = []
    for lag in range(0, max_lag + 1):
        x = grp[pol].shift(lag).dropna()
        y = grp["resp_cases"].iloc[lag:len(grp)]
        n = min(len(x), len(y))
        if n < 30:
            results.append({"Lag": lag, "Pearson": 0, "Spearman": 0, "p_value": 1})
            continue
        xv, yv = x.values[:n], y.values[:n]
        if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
            results.append({"Lag": lag, "Pearson": 0, "Spearman": 0, "p_value": 1})
            continue
        try:
            p_r, p_p = stats.pearsonr(xv, yv)
            s_r, _   = stats.spearmanr(xv, yv)
            if np.isnan(p_r): p_r = 0.0
            if np.isnan(s_r): s_r = 0.0
            if np.isnan(p_p): p_p = 1.0
        except Exception:
            p_r, s_r, p_p = 0.0, 0.0, 1.0
        results.append({
            "Lag": lag, "Pearson": round(p_r, 4),
            "Spearman": round(s_r, 4), "p_value": round(p_p, 5),
        })

    lag_df = pd.DataFrame(results)
    lag_df["Pearson"]  = lag_df["Pearson"].fillna(0)
    lag_df["Spearman"] = lag_df["Spearman"].fillna(0)
    lag_df["p_value"]  = lag_df["p_value"].fillna(1)

    best_idx = lag_df["Pearson"].abs().idxmax()
    if pd.isna(best_idx): best_idx = 0
    best = lag_df.loc[best_idx]

    sig_text = ("statistically significant (p < 0.05)"
                if best["p_value"] < 0.05 else "not statistically significant")
    sig_color = "#22c55e" if best["p_value"] < 0.05 else "#f97316"

    st.markdown(
        f"<div style='background:#0e1321;border:1px solid rgba(255,255,255,0.06);"
        f"border-left:3px solid {sig_color};border-radius:8px;padding:.9rem 1.1rem;"
        f"font-family:Outfit,sans-serif;font-size:.85rem;color:#8892a4;line-height:1.7'>"
        f"In <b style='color:#dde3f0'>{city_sel}</b>, {pol} shows its strongest "
        f"association with respiratory cases at a "
        f"<b style='color:#00b7ff'>{int(best['Lag'])}-day lag</b> "
        f"(Pearson r = <b style='color:#dde3f0'>{best['Pearson']:.3f}</b>, "
        f"Spearman r = <b style='color:#dde3f0'>{best['Spearman']:.3f}</b>, "
        f"<span style='color:{sig_color}'>{sig_text}</span>)."
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.4, 1])

    with col1:
        bar_colors = [
            "#ff3b5c" if l == int(best["Lag"]) else
            hex_to_rgba("#00b7ff", 0.7)
            for l in lag_df["Lag"]
        ]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"L{l}" for l in lag_df["Lag"]],
            y=lag_df["Pearson"],
            name="Pearson r",
            marker_color=bar_colors,
            marker_line_width=0,
        ))
        fig.add_trace(go.Scatter(
            x=[f"L{l}" for l in lag_df["Lag"]],
            y=lag_df["Spearman"],
            mode="lines+markers",
            name="Spearman r",
            line=dict(color="#ffb800", width=2.2),
            marker=dict(size=7, color="#ffb800",
                        line=dict(color="#080b12", width=1.5)),
        ))
        fig.add_hline(y=0, line_dash="dot",
                      line_color="rgba(255,255,255,0.1)")
        fig.update_layout(**layout(
            title=f"Lag Correlation: {pol} → Respiratory Cases",
            xaxis_title="Lag (days)", yaxis_title="Correlation (r)",
            height=380, title_font_size=13,
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        styled = lag_df.copy()
        styled["Sig"] = styled["p_value"].apply(
            lambda p: "***" if p < 0.001 else "**" if p < 0.01
                      else "*" if p < 0.05 else "ns"
        )
        styled[""] = styled["Lag"].apply(
            lambda l: "⭐" if l == int(best["Lag"]) else ""
        )
        st.dataframe(
            styled[["Lag","Pearson","Spearman","p_value","Sig",""]],
            use_container_width=True, height=380,
        )

    # ── Lagged scatter ──
    st.markdown(
        f"<h3 style='font-size:1rem;margin:.8rem 0'>"
        f"Lagged Scatter — {pol} at t−{lag_days} vs Cases at t</h3>",
        unsafe_allow_html=True,
    )
    x_lag = grp[pol].shift(lag_days).dropna()
    y_lag = grp["resp_cases"].iloc[lag_days:len(grp)]
    n     = min(len(x_lag), len(y_lag))
    scatter_df = pd.DataFrame({
        "Pollution": x_lag.values[:n],
        "Cases"    : y_lag.values[:n],
        "Date"     : grp["Date"].iloc[lag_days:lag_days + n].values,
    }).dropna()

    r_val, p_val = 0.0, 1.0
    x_line = y_line = np.array([])

    if len(scatter_df) > 5:
        slope, intercept, r_val, p_val, _ = stats.linregress(
            scatter_df["Pollution"], scatter_df["Cases"],
        )
        x_line = np.linspace(scatter_df["Pollution"].min(),
                              scatter_df["Pollution"].max(), 100)
        y_line = slope * x_line + intercept

    p_label = "<0.001" if p_val < 0.001 else f"{p_val:.4f}"
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=scatter_df["Pollution"], y=scatter_df["Cases"],
        mode="markers",
        marker=dict(
            color=scatter_df["Pollution"],
            colorscale=[
                [0, "#0a2540"], [0.4, "#0f4c75"],
                [0.7, "#f97316"], [1, "#ef4444"],
            ],
            size=5, opacity=0.6,
            showscale=True,
            colorbar=dict(thickness=10,
                          title=dict(text=f"{pol}", side="right")),
        ),
        text=pd.to_datetime(scatter_df["Date"]).dt.strftime("%Y-%m-%d"),
        hovertemplate=(f"{pol}: %{{x:.1f}} {unit}<br>"
                       "Cases: %{y}<br>Date: %{text}<extra></extra>"),
        name="Data",
    ))
    if len(x_line) > 0:
        fig2.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            name=f"Trend (r = {r_val:.3f})",
            line=dict(color="rgba(255,255,255,0.55)", width=2, dash="dash"),
        ))
    fig2.update_layout(**layout(
        title=(f"{city_sel} — {pol} at t−{lag_days}d vs Cases at t  "
               f"(r = {r_val:.3f}, p = {p_label})"),
        xaxis_title=f"{pol} ({unit}) — lagged {lag_days}d",
        yaxis_title="Resp. Cases / day",
        height=420, title_font_size=12,
    ))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Cross-pollutant comparison ──
    st.markdown(
        "<h3 style='font-size:1rem;margin:.8rem 0'>"
        "Cross-Pollutant Comparison at Optimal Lag</h3>",
        unsafe_allow_html=True,
    )
    pols_avail = [p for p in ["PM2.5","PM10","NO2","SO2","CO","O3"] if p in grp.columns]
    best_lags  = []
    for p in pols_avail:
        best_r, best_l = 0.0, 0
        for lag in range(0, max_lag + 1):
            x2 = grp[p].shift(lag).dropna()
            y2 = grp["resp_cases"].iloc[lag:]
            n2 = min(len(x2), len(y2))
            if n2 < 30: continue
            xv2, yv2 = x2.values[:n2], y2.values[:n2]
            if np.nanstd(xv2) == 0 or np.nanstd(yv2) == 0: continue
            try:
                r2, _ = stats.pearsonr(xv2, yv2)
                if np.isnan(r2): continue
            except Exception:
                continue
            if abs(r2) > abs(best_r):
                best_r, best_l = r2, lag
        best_lags.append({"Pollutant": p, "Best Lag": best_l,
                           "Pearson r": round(best_r, 4)})

    best_lag_df = pd.DataFrame(best_lags).sort_values("Pearson r", ascending=False)
    bar_clr = ["#22c55e" if v > 0 else "#ef4444"
               for v in best_lag_df["Pearson r"]]
    fig3 = go.Figure(go.Bar(
        x=best_lag_df["Pollutant"],
        y=best_lag_df["Pearson r"],
        marker_color=bar_clr,
        marker_line_width=0,
        text=[f"Lag {v}d" for v in best_lag_df["Best Lag"]],
        textposition="outside",
        textfont=dict(family="DM Mono, monospace", size=9, color="#8892a4"),
    ))
    fig3.update_layout(**layout(
        title=f"{city_sel} — Strongest r at Optimal Lag per Pollutant",
        height=330, title_font_size=13,
        yaxis_title="Pearson r", showlegend=False,
    ))
    st.plotly_chart(fig3, use_container_width=True)


def _tab_correlations(df, pol):
    num_cols = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3","AQI","resp_cases"]
                if c in df.columns and df[c].notna().sum() > 100]

    col1, col2 = st.columns(2)
    for col_idx, method in enumerate(["pearson","spearman"]):
        corr = df[num_cols].corr(method=method).round(3)
        with [col1, col2][col_idx]:
            fig = px.imshow(
                corr,
                color_continuous_scale=[
                    [0.00, "#7f1d1d"], [0.25, "#0e1321"],
                    [0.50, "#111827"], [0.75, "#14532d"],
                    [1.00, "#052e16"],
                ],
                zmid=0, zmin=-1, zmax=1,
                text_auto=True,
                title=f"{method.capitalize()} Correlation",
                aspect="auto",
            )
            fig.update_layout(**layout(
                height=400, title_font_size=13,
                coloraxis_colorbar=dict(thickness=12, len=0.85),
            ))
            fig.update_traces(
                textfont=dict(size=9.5, family="DM Mono, monospace"),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Feature ranking ──
    st.markdown(
        "<h3 style='font-size:1rem;margin:.4rem 0 .8rem'>"
        "Feature Correlation with Respiratory Cases (Ranked)</h3>",
        unsafe_allow_html=True,
    )
    feature_cols = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3","AQI"]
                    if c in df.columns and df[c].notna().sum() > 100]
    rows = []
    for f in feature_cols:
        sub = df[[f,"resp_cases"]].dropna()
        if len(sub) < 30: continue
        p_r, p_p = stats.pearsonr(sub[f], sub["resp_cases"])
        s_r, s_p = stats.spearmanr(sub[f], sub["resp_cases"])
        rows.append({
            "Feature": f,
            "Pearson r": round(p_r, 4),
            "Spearman r": round(s_r, 4),
            "p-value": round(p_p, 5),
            "Sig": ("***" if p_p < 0.001 else "**" if p_p < 0.01
                    else "*" if p_p < 0.05 else "ns"),
        })
    corr_rank = pd.DataFrame(rows).sort_values("Pearson r", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=corr_rank["Feature"], y=corr_rank["Pearson r"],
        name="Pearson r",
        marker_color=COLORS[:len(corr_rank)],
        marker_line_width=0,
    ))
    fig.add_trace(go.Scatter(
        x=corr_rank["Feature"], y=corr_rank["Spearman r"],
        name="Spearman r", mode="lines+markers",
        line=dict(color="#ffb800", width=2.2),
        marker=dict(size=8, color="#ffb800",
                    line=dict(color="#080b12", width=1.5)),
    ))
    fig.update_layout(**layout(
        title="Pollutant → Respiratory Cases — Pearson & Spearman",
        xaxis_title="", yaxis_title="r",
        height=340, title_font_size=13,
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(corr_rank, use_container_width=True)

    # ── Scatter matrix ──
    st.markdown(
        "<h3 style='font-size:1rem;margin:.8rem 0'>Scatter Matrix</h3>",
        unsafe_allow_html=True,
    )
    sample = df[feature_cols + ["resp_cases","City"]].dropna().sample(
        min(2000, len(df)), random_state=42,
    )
    fig2 = px.scatter_matrix(
        sample, dimensions=feature_cols[:5],
        color="City", color_discrete_sequence=COLORS,
        opacity=0.35,
        title="Scatter Matrix — Key Pollutants",
    )
    fig2.update_traces(diagonal_visible=False, marker_size=2.5)
    fig2.update_layout(**layout(height=560, title_font_size=13))
    st.plotly_chart(fig2, use_container_width=True)


def _tab_cities(df, pols, cities):
    st.markdown(
        "<h3 style='font-size:1rem;margin:.2rem 0 .8rem'>"
        "City-wise Pollutant Averages</h3>",
        unsafe_allow_html=True,
    )

    city_avg = df.groupby("City")[pols].mean().round(2).reset_index()
    fig = go.Figure()
    for i, p in enumerate(pols):
        fig.add_trace(go.Bar(
            x=city_avg["City"], y=city_avg[p],
            name=p, marker_color=COLORS[i],
            marker_line_width=0,
        ))
    fig.update_layout(**layout(
        barmode="group",
        title="City-wise Average Pollutants",
        xaxis_title="", yaxis_title="Concentration",
        height=380,
    ))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig2 = px.violin(
            df, x="City", y="AQI", color="City",
            color_discrete_sequence=COLORS,
            box=True, points=False,
            title="AQI Distribution by City",
        )
        fig2.update_layout(**layout(height=380, title_font_size=13, showlegend=False))
        fig2.update_xaxes(tickangle=-30)
        fig2.update_traces(meanline_visible=True)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        bucket_city = (df.groupby(["City","AQI_Bucket"])
                         .size().reset_index(name="count"))
        total = bucket_city.groupby("City")["count"].transform("sum")
        bucket_city["pct"] = (bucket_city["count"] / total * 100).round(1)
        bcolors = {k: v[2] for k, v in AQI_BUCKETS.items()}
        fig3 = px.bar(
            bucket_city, x="City", y="pct",
            color="AQI_Bucket", color_discrete_map=bcolors,
            title="AQI Bucket Share by City (%)",
            labels={"pct": "%"},
        )
        fig3.update_layout(**layout(height=380, title_font_size=13, barmode="stack"))
        fig3.update_xaxes(tickangle=-30)
        fig3.update_traces(marker_line_width=0)
        st.plotly_chart(fig3, use_container_width=True)

    # ── City × Year ──
    st.markdown(
        "<h3 style='font-size:1rem;margin:.8rem 0'>City × Year Annual Average</h3>",
        unsafe_allow_html=True,
    )
    pol_sel = st.selectbox("Pollutant", pols, key="city_year_pol")
    cy = df.groupby(["City","Year"])[pol_sel].mean().round(2).reset_index()
    fig4 = px.line(
        cy, x="Year", y=pol_sel, color="City",
        color_discrete_sequence=COLORS, markers=True,
        title=f"Annual Average {pol_sel} by City",
    )
    fig4.update_layout(**layout(
        height=380, title_font_size=13,
        xaxis_title="Year", yaxis_title=f"{pol_sel} ({_pollutant_unit(pol_sel)})",
        xaxis=dict(dtick=1),
    ))
    fig4.update_traces(marker_size=7, line_width=2)
    st.plotly_chart(fig4, use_container_width=True)

    # ── Radar ──
    st.markdown(
        "<h3 style='font-size:1rem;margin:.8rem 0'>Radar — Normalised Fingerprint per City</h3>",
        unsafe_allow_html=True,
    )
    radar_pols = [p for p in ["PM2.5","PM10","NO2","SO2","CO","O3"] if p in df.columns]
    city_avg2  = df.groupby("City")[radar_pols].mean()
    norm = (city_avg2 - city_avg2.min()) / (city_avg2.max() - city_avg2.min() + 1e-9)
    fig5 = go.Figure()
    for i, city in enumerate(norm.index[:8]):
        vals = norm.loc[city].tolist()
        clr  = COLORS[i % len(COLORS)]
        fig5.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_pols + [radar_pols[0]],
            fill="toself",
            name=city,
            line=dict(color=clr, width=2),
            fillcolor=hex_to_rgba(clr, 0.12),
        ))
    fig5.update_layout(**layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(color="#2d3a4f", size=9,
                              family="DM Mono, monospace"),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=10.5, color="#8892a4",
                              family="Outfit, sans-serif"),
            ),
        ),
        height=480,
        title="Normalised Pollutant Fingerprint (0 – 1)",
        showlegend=True,
    ))
    st.plotly_chart(fig5, use_container_width=True)


def _tab_eda(df, pol):
    unit = _pollutant_unit(pol)

    # ── Outlier detection ──
    st.markdown(
        "<h3 style='font-size:1rem;margin:.2rem 0 .6rem'>"
        "Outlier Detection — Z-Score Method</h3>",
        unsafe_allow_html=True,
    )
    city_sel = st.selectbox("City", sorted(df["City"].unique()), key="eda_city")
    grp = df[df["City"] == city_sel].sort_values("Date").reset_index(drop=True)

    z_scores     = np.abs(stats.zscore(grp[pol].fillna(grp[pol].median())))
    outlier_mask = z_scores > 2.5
    outliers     = grp[outlier_mask]

    pct_out = outlier_mask.mean() * 100
    st.markdown(
        f"<div style='font-family:DM Mono,monospace;font-size:.72rem;color:#4a5568;"
        f"margin-bottom:.6rem'>"
        f"Detected <b style='color:#ff3b5c'>{outlier_mask.sum()} outlier days</b> "
        f"({pct_out:.1f}%) with |z| > 2.5 out of {len(grp):,} total days in "
        f"<b style='color:#dde3f0'>{city_sel}</b>.</div>",
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grp["Date"], y=grp[pol],
        mode="lines", name="Daily",
        line=dict(color="rgba(0,183,255,0.2)", width=1),
        fill="tozeroy", fillcolor="rgba(0,183,255,0.03)",
    ))
    fig.add_trace(go.Scatter(
        x=grp["Date"], y=grp[pol].rolling(14, min_periods=1).mean(),
        mode="lines", name="14d MA", line=dict(color="#00b7ff", width=2),
    ))
    if not outliers.empty:
        fig.add_trace(go.Scatter(
            x=outliers["Date"], y=outliers[pol],
            mode="markers", name="Outlier (|z| > 2.5)",
            marker=dict(color="#ff3b5c", size=8, symbol="x",
                        line=dict(width=2, color="#ff3b5c")),
            text=outliers["Date"].dt.strftime("%Y-%m-%d"),
            hovertemplate=(f"{pol}: %{{y:.1f}} {unit}<br>"
                           "Date: %{text}<extra></extra>"),
        ))
    fig.update_layout(**layout(
        title=f"{city_sel} — {pol} with Outliers Highlighted",
        xaxis_title="", yaxis_title=f"{pol} ({unit})", height=380,
    ))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        vals = grp[pol].dropna()
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=vals, nbinsx=50,
            marker_color="rgba(0,183,255,0.5)",
            marker_line_color="rgba(0,183,255,0.2)",
            marker_line_width=0.5,
            histnorm="probability density",
            name="Density",
        ))
        kde   = stats.gaussian_kde(vals)
        x_kde = np.linspace(vals.min(), vals.max(), 200)
        fig2.add_trace(go.Scatter(
            x=x_kde, y=kde(x_kde),
            mode="lines", name="KDE",
            line=dict(color="#ff3b5c", width=2.5),
        ))
        fig2.add_vline(x=vals.mean(), line_color="#ffb800", line_dash="dash",
                       annotation_text=f"Mean {vals.mean():.1f}",
                       annotation_font_color="#ffb800",
                       annotation_font_family="DM Mono, monospace",
                       annotation_font_size=10)
        fig2.add_vline(x=vals.median(), line_color="#22c55e", line_dash="dot",
                       annotation_text=f"Median {vals.median():.1f}",
                       annotation_font_color="#22c55e",
                       annotation_font_family="DM Mono, monospace",
                       annotation_font_size=10)
        fig2.update_layout(**layout(
            title=f"{city_sel} — {pol} Distribution",
            xaxis_title=f"{pol} ({unit})", yaxis_title="Density",
            height=340, title_font_size=13,
        ))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.box(
            grp, x="Season", y=pol, color="Season",
            color_discrete_map=SEASON_COLORS,
            title=f"{city_sel} — {pol} by Season",
            category_orders={"Season": ["Winter","Spring","Summer","Autumn"]},
            points=False,
        )
        fig3.update_layout(**layout(height=340, title_font_size=13, showlegend=False))
        fig3.update_traces(line_color="rgba(255,255,255,0.3)", fillcolor=None)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Year-over-year box ──
    st.markdown(
        "<h3 style='font-size:1rem;margin:.8rem 0'>Year-over-Year Distribution</h3>",
        unsafe_allow_html=True,
    )
    fig4 = px.box(
        grp, x="Year", y=pol, color="Year",
        color_discrete_sequence=COLORS,
        title=f"{city_sel} — {pol} Year-over-Year",
        points=False,
    )
    fig4.update_layout(**layout(height=360, title_font_size=13, showlegend=False))
    st.plotly_chart(fig4, use_container_width=True)

    # ── Descriptive stats ──
    st.markdown(
        "<h3 style='font-size:1rem;margin:.8rem 0'>Descriptive Statistics</h3>",
        unsafe_allow_html=True,
    )
    pols_avail = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3","AQI"] if c in grp.columns]
    desc = grp[pols_avail].describe().T.round(2)
    desc["skewness"] = grp[pols_avail].skew().round(3)
    desc["kurtosis"] = grp[pols_avail].kurtosis().round(3)
    st.dataframe(desc, use_container_width=True)

    # ── Q-Q plot ──
    st.markdown(
        "<h3 style='font-size:1rem;margin:.8rem 0'>Q-Q Plot (Normality Check)</h3>",
        unsafe_allow_html=True,
    )
    qq_pol  = st.selectbox("Pollutant", pols_avail, key="qq_pol")
    qq_vals = grp[qq_pol].dropna()
    (osm, osr), (slope, intercept, r) = stats.probplot(qq_vals)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers",
        marker=dict(color="#00b7ff", size=4, opacity=0.55,
                    line=dict(width=0)),
        name="Sample quantiles",
    ))
    x_line2 = np.array([min(osm), max(osm)])
    fig5.add_trace(go.Scatter(
        x=x_line2, y=slope * x_line2 + intercept,
        mode="lines", name="Normal reference",
        line=dict(color="#ff3b5c", width=2.5),
    ))
    fig5.update_layout(**layout(
        title=f"{city_sel} — {qq_pol} Q-Q Plot  (R² = {r**2:.3f})",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=360, title_font_size=13,
    ))
    st.plotly_chart(fig5, use_container_width=True)

    if r**2 > 0.98:
        st.success(f"✅ {qq_pol} is approximately normally distributed (R² = {r**2:.3f})")
    else:
        st.warning(
            f"⚠ {qq_pol} deviates from normality (R² = {r**2:.3f}) "
            "— Spearman correlation is preferred."
        )


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

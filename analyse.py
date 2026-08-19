"""
Commercial & General Data Analytics Dashboard
------------------------------------------------
A drag-and-drop Streamlit app that works two ways:

  - FMCG / Retail mode: auto-detects commercial columns (volume, revenue,
    retail account, campaign, stock) the way the original version did.
  - General Data mode: works on ANY csv/xlsx file — you confirm which
    column is the date, which is the main metric, and which is the
    grouping/category, and the whole dashboard adapts to that.

Either way you get: cleaned data, KPIs, written analyst-style findings,
interactive charts, a natural-language "Ask Your Data" chat (free, via
Groq), a PDF export of the executive summary, and a CSV download of the
cleaned data.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    Push app.py + requirements.txt to a GitHub repo, then create the app
    at https://share.streamlit.io pointing at app.py. Add your free Groq
    key as a secret named GROQ_API_KEY so colleagues don't need their own.
"""

import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(page_title="Data Analytics Dashboard", page_icon="📊", layout="wide")
st.title("📊 Data Analytics Dashboard")
st.caption(
    "Drop a CSV or Excel file below. The app cleans it, builds a full "
    "dashboard, and writes up the key findings the way a senior analyst would."
)

# ----------------------------------------------------------------------
# PASSWORD PROTECTION — simple shared-password gate for an internal tool.
# Set APP_PASSWORD in .streamlit/secrets.toml (local) or Streamlit Cloud
# secrets (deployed). Nothing below this runs until the password matches.
# ----------------------------------------------------------------------
def check_password():
    try:
        configured_password = st.secrets.get("APP_PASSWORD", None)
    except Exception:
        configured_password = None

    if configured_password is None:
        return True  # no password configured — app stays open

    def password_entered():
        if st.session_state.get("password") == configured_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("🔒 Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()

# ----------------------------------------------------------------------
# COLUMN ALIASES — used in FMCG mode to recognize commercial columns
# ----------------------------------------------------------------------
COLUMN_ALIASES = {
    "volume": ["volume", "units", "unit_sales", "qty", "quantity", "cases", "volume_units"],
    "revenue": ["revenue", "sales", "sales_value", "net_revenue", "turnover", "value", "amount"],
    "account_name": [
        "account_name", "account", "retailer", "retail_partner", "customer",
        "customer_name", "store", "outlet", "partner_name", "key_account",
    ],
    "date": ["date", "period", "transaction_date", "order_date", "week", "month", "day"],
    "stock": ["stock", "stock_allocation", "inventory", "allocated_stock", "stock_qty"],
    "campaign": ["campaign", "campaign_name", "promo", "promotion", "campaign_id"],
}


# ----------------------------------------------------------------------
# CLEANING PIPELINE (runs the same way regardless of mode)
# ----------------------------------------------------------------------
def standardize_column_name(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^\w]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")


def map_to_canonical(columns):
    """Rename exact alias matches (e.g. a column literally called 'revenue')
    to canonical names. Harmless on non-commercial data — it only fires on
    exact name matches, never guesses based on values."""
    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for col in columns:
            if col in aliases and col not in mapping:
                mapping[col] = canonical
                break
    return mapping


@st.cache_data(show_spinner=False)
def load_raw_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    if file_name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(buffer)
        except UnicodeDecodeError:
            buffer.seek(0)
            df = pd.read_csv(buffer, encoding="latin1")
    else:
        df = pd.read_excel(buffer)
    return df


def detect_date_columns(df: pd.DataFrame, threshold: float = 0.7) -> list:
    candidates = []
    for col in df.columns:
        name_hint = any(k in col for k in ["date", "period", "month", "week", "day", "time"])
        if df[col].dtype == object or name_hint:
            sample = df[col].dropna().astype(str).head(50)
            if len(sample) == 0:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", format=None)
            success_rate = parsed.notna().mean()
            if name_hint or success_rate >= threshold:
                candidates.append(col)
    return candidates


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [standardize_column_name(c) for c in df.columns]
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    mapping = map_to_canonical(df.columns)
    if mapping:
        df = df.rename(columns=mapping)

    date_cols = detect_date_columns(df)
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    if "date" not in df.columns and date_cols:
        df["date"] = df[date_cols[0]]

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if "pct" in col or "percent" in col or "rate" in col:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        else:
            df[col] = df[col].fillna("Unknown")
            df[col] = df[col].astype(str).str.strip()

    df = df.drop_duplicates()
    return df


def find_first_match(df, canonical_name, fallback_keywords):
    if canonical_name in df.columns:
        return canonical_name
    for col in df.columns:
        if any(k in col for k in fallback_keywords):
            return col
    return None


def fmt_num(x, prefix="", suffix=""):
    if pd.isna(x):
        return "N/A"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000:
        return f"{sign}{prefix}{x/1_000_000:.2f}M{suffix}"
    if x >= 1_000:
        return f"{sign}{prefix}{x/1_000:.1f}K{suffix}"
    return f"{sign}{prefix}{x:,.0f}{suffix}"


def label_for(col):
    return col.replace("_", " ").title() if col else None


def opt_index(options, value):
    """Index of value in options, defaulting to 0 if not found/None."""
    try:
        return options.index(value)
    except (ValueError, TypeError):
        return 0


# ----------------------------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader("📁 Drop your CSV or Excel file here", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("⬆️ Waiting for a file... Upload any CSV or Excel export.")
    st.stop()

with st.spinner("Reading file..."):
    try:
        raw_df = load_raw_file(uploaded_file.getvalue(), uploaded_file.name)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

if raw_df.empty:
    st.error("The uploaded file appears to be empty.")
    st.stop()

with st.spinner("Cleaning and standardizing data..."):
    df_full = clean_data(raw_df)

st.success(f"File loaded — {df_full.shape[0]:,} rows × {df_full.shape[1]} columns after cleaning.")

# ----------------------------------------------------------------------
# ANALYSIS MODE
# ----------------------------------------------------------------------
st.sidebar.header("📊 Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Data type", ["FMCG / Retail", "General Data"], index=0,
    help="FMCG mode auto-detects commercial columns (revenue, accounts, campaigns, "
         "stock). General mode works on any dataset — you'll confirm your columns below.",
)
is_fmcg_mode = analysis_mode == "FMCG / Retail"

# ----------------------------------------------------------------------
# COLUMN AUTO-DETECTION (mode-aware best guesses)
# ----------------------------------------------------------------------
numeric_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
text_cols = df_full.select_dtypes(include=["object"]).columns.tolist()
date_options = df_full.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

if is_fmcg_mode:
    volume_guess = find_first_match(df_full, "volume", ["volume", "unit", "qty", "quantity", "case"])
    revenue_guess = find_first_match(df_full, "revenue", ["revenue", "sales", "turnover", "value", "amount"])
    account_guess = find_first_match(df_full, "account_name", ["account", "retail", "customer", "store", "outlet", "partner"])
    date_guess = find_first_match(df_full, "date", ["date", "period", "month", "week"])
    stock_guess = find_first_match(df_full, "stock", ["stock", "inventory", "allocat"])
    campaign_guess = find_first_match(df_full, "campaign", ["campaign", "promo"])

    metric_guess = revenue_guess or volume_guess or (numeric_cols[0] if numeric_cols else None)
    secondary_metric_guess = stock_guess or (volume_guess if volume_guess != metric_guess else None)
    category_guess = account_guess
    segment_guess = campaign_guess
else:
    date_guess = date_options[0] if date_options else None
    metric_guess = numeric_cols[0] if numeric_cols else None
    remaining_numeric = [c for c in numeric_cols if c != metric_guess]
    secondary_metric_guess = remaining_numeric[0] if remaining_numeric else None
    category_guess = text_cols[0] if text_cols else None
    remaining_text = [c for c in text_cols if c != category_guess]
    segment_guess = remaining_text[0] if remaining_text else None

# ----------------------------------------------------------------------
# CONFIRM DATA SETUP — 3 quick questions, both modes, smart defaults
# ----------------------------------------------------------------------
st.markdown("### ⚙️ Confirm what your columns mean")
st.caption("Auto-detected below — change anything that's wrong before the dashboard builds.")

date_choices = ["(none)"] + date_options
metric_choices = numeric_cols if numeric_cols else ["(none)"]
category_choices = ["(none)"] + text_cols

c1, c2, c3 = st.columns(3)
with c1:
    date_pick = st.selectbox("📅 Date / time column", date_choices, index=opt_index(date_choices, date_guess))
with c2:
    metric_pick = st.selectbox("🔢 Main metric to analyze", metric_choices, index=opt_index(metric_choices, metric_guess))
with c3:
    category_pick = st.selectbox("🏷️ Main grouping / category column", category_choices, index=opt_index(category_choices, category_guess))

date_col = None if date_pick == "(none)" else date_pick
metric_col = None if metric_pick == "(none)" else metric_pick
category_col = None if category_pick == "(none)" else category_pick

with st.expander("Advanced (optional): secondary metric & secondary grouping"):
    ac1, ac2 = st.columns(2)
    secondary_choices = ["(none)"] + [c for c in numeric_cols if c != metric_col]
    segment_choices = ["(none)"] + [c for c in text_cols if c != category_col]
    with ac1:
        secondary_pick = st.selectbox(
            "Secondary metric (e.g. stock, cost, headcount)", secondary_choices,
            index=opt_index(secondary_choices, secondary_metric_guess),
        )
    with ac2:
        segment_pick = st.selectbox(
            "Secondary grouping (e.g. campaign, channel, region)", segment_choices,
            index=opt_index(segment_choices, segment_guess),
        )
    secondary_metric_col = None if secondary_pick == "(none)" else secondary_pick
    segment_col = None if segment_pick == "(none)" else segment_pick

metric_label = label_for(metric_col) or "Metric"
category_label = label_for(category_col) or "Category"
segment_label = label_for(segment_col) or "Segment"
secondary_metric_label = label_for(secondary_metric_col) or "Secondary Metric"

st.divider()

# ----------------------------------------------------------------------
# SIDEBAR FILTERS (apply globally to every tab)
# ----------------------------------------------------------------------
st.sidebar.header("🔎 Filters")
df = df_full.copy()

if date_col and df[date_col].notna().any():
    min_date, max_date = df[date_col].min().date(), df[date_col].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        df = df[(df[date_col].dt.date >= start) & (df[date_col].dt.date <= end)]

if category_col:
    all_categories = sorted(df_full[category_col].unique().tolist())
    selected_categories = st.sidebar.multiselect(f"{category_label}", all_categories, default=all_categories)
    df = df[df[category_col].isin(selected_categories)]

if segment_col:
    all_segments = sorted(df_full[segment_col].unique().tolist())
    selected_segments = st.sidebar.multiselect(f"{segment_label}", all_segments, default=all_segments)
    df = df[df[segment_col].isin(selected_segments)]

if df.empty:
    st.warning("No rows match the current filters — widen your selection in the sidebar.")
    st.stop()

st.sidebar.caption(f"Showing {len(df):,} of {len(df_full):,} rows after filters.")

# ----------------------------------------------------------------------
# GROQ API SETUP — powers "Ask Your Data", free tier, no card needed
# ----------------------------------------------------------------------
st.sidebar.header("🤖 Ask Your Data")
_default_key = ""
try:
    _default_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    _default_key = ""

groq_api_key = st.sidebar.text_input(
    "Groq API key (free)", value=_default_key, type="password",
    help=(
        "Get a free key at console.groq.com — no credit card required. "
        "Not saved anywhere, only kept for this session. For a deployed app, "
        "add it as a Streamlit Cloud secret named GROQ_API_KEY instead."
    ),
)
groq_model = st.sidebar.selectbox(
    "Model", ["openai/gpt-oss-20b", "openai/gpt-oss-120b"], index=0,
    help="20B is faster with the most generous free daily limit. 120B is smarter for tricky questions.",
)

# ----------------------------------------------------------------------
# ANALYSIS HELPERS
# ----------------------------------------------------------------------
def period_over_period(df, metric_col, date_col, freq="W"):
    if not (date_col and metric_col):
        return None, None
    ts = df.dropna(subset=[date_col]).set_index(date_col)[metric_col].resample(freq).sum().sort_index()
    ts = ts[ts.index.notna()]
    if len(ts) < 2 or ts.iloc[-2] == 0:
        return ts, None
    pct = (ts.iloc[-1] - ts.iloc[-2]) / ts.iloc[-2] * 100
    return ts, pct


def category_growth_table(df, category_col, metric_col, date_col):
    if not (category_col and metric_col and date_col):
        return None
    valid = df.dropna(subset=[date_col])
    if valid[date_col].nunique() < 2:
        return None
    midpoint = valid[date_col].median()
    first_half = valid[valid[date_col] <= midpoint].groupby(category_col)[metric_col].sum()
    second_half = valid[valid[date_col] > midpoint].groupby(category_col)[metric_col].sum()
    table = pd.DataFrame({"earlier_period": first_half, "later_period": second_half}).fillna(0)
    table["change_pct"] = np.where(
        table["earlier_period"] != 0,
        (table["later_period"] - table["earlier_period"]) / table["earlier_period"] * 100,
        np.nan,
    )
    return table.sort_values("later_period", ascending=False)


def linear_forecast(ts: pd.Series, periods_ahead: int = 4):
    if ts is None or len(ts) < 4:
        return None, None
    y = ts.values.astype(float)
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)
    trend = np.poly1d(coeffs)
    future_x = np.arange(len(y), len(y) + periods_ahead)
    future_y = np.clip(trend(future_x), 0, None)
    freq = ts.index.freq or pd.infer_freq(ts.index)
    future_index = pd.date_range(ts.index[-1], periods=periods_ahead + 1, freq=freq or "W")[1:]
    forecast = pd.Series(future_y, index=future_index)
    direction = "upward" if coeffs[0] > 0 else "downward" if coeffs[0] < 0 else "flat"
    return forecast, direction


def generate_executive_insights(df, category_col, category_label, metric_col, metric_label,
                                 date_col, segment_col, segment_label,
                                 secondary_metric_col, secondary_metric_label):
    insights = []

    if category_col and metric_col:
        by_cat = df.groupby(category_col)[metric_col].sum().sort_values(ascending=False)
        total = by_cat.sum()
        if len(by_cat) > 0 and total > 0:
            top_name, top_val = by_cat.index[0], by_cat.iloc[0]
            top_share = top_val / total * 100
            insights.append(
                f"**{top_name}** leads on {category_label.lower()}, contributing **{fmt_num(top_val)}** "
                f"in {metric_label.lower()} — **{top_share:.1f}%** of the total across all "
                f"{len(by_cat)} {category_label.lower()} groups."
            )
            if len(by_cat) > 1:
                bottom_name, bottom_val = by_cat.index[-1], by_cat.iloc[-1]
                if bottom_val > 0:
                    insights.append(
                        f"**{bottom_name}** is the weakest at **{fmt_num(bottom_val)}**, roughly "
                        f"**{(top_val / bottom_val):.1f}x** below the top performer."
                    )

    ts, pct = period_over_period(df, metric_col, date_col, freq="W")
    if pct is not None:
        direction = "grew" if pct >= 0 else "declined"
        insights.append(f"The most recent period-over-period {metric_label.lower()} trend **{direction} by {abs(pct):.1f}%**.")

    growth_table = category_growth_table(df, category_col, metric_col, date_col)
    if growth_table is not None and not growth_table["change_pct"].isna().all():
        gt = growth_table.dropna(subset=["change_pct"])
        if not gt.empty:
            best, worst = gt["change_pct"].idxmax(), gt["change_pct"].idxmin()
            insights.append(
                f"Comparing the first and second half of the selected period, **{best}** posted the "
                f"strongest growth (**{gt.loc[best, 'change_pct']:+.1f}%**), while **{worst}** saw the "
                f"largest decline (**{gt.loc[worst, 'change_pct']:+.1f}%**)."
            )

    if segment_col and metric_col:
        by_seg = df.groupby(segment_col)[metric_col].mean().sort_values(ascending=False)
        if len(by_seg) > 1:
            insights.append(
                f"**{by_seg.index[0]}** shows the highest average {metric_label.lower()} per record "
                f"among {segment_label.lower()} groups — a candidate to prioritize or extend."
            )

    if secondary_metric_col and category_col:
        ratio_df = df.groupby(category_col)[[metric_col, secondary_metric_col]].sum()
        if (ratio_df[secondary_metric_col] > 0).any():
            ratio_df["ratio"] = ratio_df[metric_col] / ratio_df[secondary_metric_col].replace(0, np.nan)
            high = ratio_df[ratio_df["ratio"] > ratio_df["ratio"].quantile(0.85)]
            if not high.empty:
                names = ", ".join(high.index.tolist()[:3])
                insights.append(
                    f"⚠️ **{names}** show an unusually high {metric_label.lower()}-to-"
                    f"{secondary_metric_label.lower()} ratio — worth checking whether "
                    f"{secondary_metric_label.lower()} is keeping pace."
                )

    if not insights:
        insights.append("Not enough recognizable columns were confirmed above to generate detailed findings.")

    return insights


# ----------------------------------------------------------------------
# PDF EXECUTIVE SUMMARY EXPORT
# ----------------------------------------------------------------------
def _pdf_safe(text: str) -> str:
    text = text.replace("**", "")
    out = []
    for ch in text:
        if ch == "€":
            out.append(ch)
        else:
            out.append(ch if ch.encode("latin-1", "ignore") else "")
    return "".join(out)


def generate_pdf_report(kpi_values, insights, ts, metric_label):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Data Analytics Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Key Performance Indicators", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for label, value in kpi_values:
        pdf.cell(0, 7, _pdf_safe(f"{label}: {value}"), ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Analyst Findings", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for point in insights:
        pdf.multi_cell(0, 6, _pdf_safe(f"- {point}"))
    pdf.ln(2)

    if ts is not None and len(ts) > 0:
        try:
            fig = px.area(x=ts.index, y=ts.values, title=f"{metric_label} Trend")
            fig.update_layout(width=900, height=380)
            img_bytes = fig.to_image(format="png", scale=2)
            tmp_path = "/tmp/_report_chart.png"
            with open(tmp_path, "wb") as f:
                f.write(img_bytes)
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "Trend", ln=True)
            pdf.image(tmp_path, w=180)
        except Exception:
            pass  # kaleido missing or export failed — skip the image, keep the report

    return bytes(pdf.output())


# ----------------------------------------------------------------------
# "ASK YOUR DATA" — natural-language Q&A over the uploaded dataset
# ----------------------------------------------------------------------
FORBIDDEN_PATTERNS = [
    "import", "open(", "exec(", "eval(", "__", "os.", "sys.", "subprocess",
    "requests", "socket", "globals", "locals", "getattr", "setattr",
    "delattr", "compile(", "input(", "write", "delete", "remove",
]
SAFE_BUILTINS = {
    "len": len, "range": range, "sum": sum, "min": min, "max": max,
    "sorted": sorted, "abs": abs, "round": round, "str": str, "int": int,
    "float": float, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "enumerate": enumerate, "zip": zip, "True": True, "False": False, "None": None,
}


def get_schema_summary(df: pd.DataFrame) -> str:
    lines = ["Columns available in `df` (already cleaned):"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        if pd.api.types.is_numeric_dtype(df[col]):
            sample = f"range {df[col].min():.2f} to {df[col].max():.2f}"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            sample = f"range {df[col].min().date()} to {df[col].max().date()}"
        else:
            uniques = df[col].dropna().unique()[:5]
            sample = f"e.g. {', '.join(str(u) for u in uniques)}"
        lines.append(f"- `{col}` ({dtype}): {sample}")
    lines.append(f"\nTotal rows: {len(df)}")
    return "\n".join(lines)


def is_code_safe(code: str) -> bool:
    lowered = code.lower()
    return not any(pattern in lowered for pattern in FORBIDDEN_PATTERNS)


def generate_pandas_code(client, model, question, schema_summary):
    system_prompt = (
        "You are a data analyst assistant. You write a single short pandas "
        "snippet that answers the user's question using a dataframe called "
        "`df` that already exists. Rules:\n"
        "- Only use `df`, `pd`, and `np` — no imports, no file/network access.\n"
        "- Assign the final answer to a variable called `result`.\n"
        "- `result` should be a DataFrame, Series, or simple scalar (number/string).\n"
        "- Keep it to a few lines. No comments, no explanation.\n"
        "- Respond with ONLY the code, no markdown fences, no prose.\n\n"
        f"{schema_summary}"
    )
    response = client.chat.completions.create(
        model=model, max_tokens=400,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
    )
    code = response.choices[0].message.content.strip()
    return code.replace("```python", "").replace("```", "").strip()


def run_pandas_code(code: str, df: pd.DataFrame):
    namespace = {"df": df.copy(), "pd": pd, "np": np, "result": None, "__builtins__": SAFE_BUILTINS}
    exec(code, namespace)  # noqa: S102 — restricted namespace + keyword scan above
    return namespace.get("result")


def summarize_result_for_model(result) -> str:
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.head(20).to_string()
    return str(result)


def generate_final_answer(client, model, question, result_text):
    system_prompt = (
        "You are a senior data analyst speaking to a non-technical colleague. "
        "You are given a question and the computed result. Write a short, "
        "direct, plain-language answer (2-4 sentences max). Use actual numbers "
        "from the result. No jargon, no code, no markdown headers."
    )
    user_msg = f"Question: {question}\n\nComputed result:\n{result_text}"
    response = client.chat.completions.create(
        model=model, max_tokens=300,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
    )
    return response.choices[0].message.content.strip()


def answer_question(question, df, api_key, model):
    client = Groq(api_key=api_key)
    schema_summary = get_schema_summary(df)

    code = generate_pandas_code(client, model, question, schema_summary)
    if not is_code_safe(code):
        return {"answer": "I can't run that request — it touched something outside the data itself.",
                "code": code, "result": None, "error": "blocked"}

    try:
        result = run_pandas_code(code, df)
    except Exception as e:
        return {"answer": f"I couldn't compute that — the generated calculation failed ({e}). Try rephrasing.",
                "code": code, "result": None, "error": str(e)}

    result_text = summarize_result_for_model(result)
    try:
        final_answer = generate_final_answer(client, model, question, result_text)
    except Exception as e:
        final_answer = f"Computed the result, but couldn't phrase the answer ({e}). Raw result: {result_text}"

    return {"answer": final_answer, "code": code, "result": result, "error": None}


# ----------------------------------------------------------------------
# TABS — built dynamically based on which columns are actually confirmed
# ----------------------------------------------------------------------
tab_labels = ["📋 Executive Summary", "🤖 Ask Your Data"]
if category_col:
    tab_labels.append(f"📊 {category_label} Performance")
if segment_col:
    tab_labels.append(f"🎯 {segment_label} Analysis")
if date_col:
    tab_labels.append("📈 Trends & Forecast")
if secondary_metric_col:
    tab_labels.append(f"📦 {secondary_metric_label} Analysis")
tab_labels.append("🔍 Data Explorer")

tabs = st.tabs(tab_labels)
_iter = iter(tabs)
tab_summary = next(_iter)
tab_ask = next(_iter)
tab_category = next(_iter) if category_col else None
tab_segment = next(_iter) if segment_col else None
tab_trends = next(_iter) if date_col else None
tab_secondary = next(_iter) if secondary_metric_col else None
tab_explorer = next(_iter)

# ---------------------- TAB: EXECUTIVE SUMMARY ----------------------
with tab_summary:
    st.markdown("### Key Performance Indicators")
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric(f"Total {metric_label}", fmt_num(df[metric_col].sum()) if metric_col else "N/A")
    with kpi_cols[1]:
        st.metric(
            f"Total {secondary_metric_label}" if secondary_metric_col else "Records",
            fmt_num(df[secondary_metric_col].sum()) if secondary_metric_col else f"{len(df):,}",
        )
    with kpi_cols[2]:
        st.metric(f"Active {category_label}s" if category_col else "Rows",
                   f"{df[category_col].nunique():,}" if category_col else f"{len(df):,}")
    with kpi_cols[3]:
        _, pct = period_over_period(df, metric_col, date_col, freq="W")
        st.metric("Latest Period-over-Period Growth", f"{pct:+.1f}%" if pct is not None else "N/A")

    st.divider()
    st.markdown("### 🧠 Analyst Findings")
    st.caption("Auto-generated directly from the filtered data above — not templated text.")
    insights = generate_executive_insights(
        df, category_col, category_label, metric_col, metric_label, date_col,
        segment_col, segment_label, secondary_metric_col, secondary_metric_label,
    )
    for point in insights:
        st.markdown(f"- {point}")

    st.divider()
    st.markdown("### Overview")
    ts, _ = period_over_period(df, metric_col, date_col, freq="W")
    if ts is not None and len(ts) > 0:
        fig = px.area(x=ts.index, y=ts.values, title=f"{metric_label} Trend (Weekly)", labels={"x": "Date", "y": metric_label})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No date column confirmed — trend overview unavailable.")

    st.divider()
    kpi_values = [
        (f"Total {metric_label}", fmt_num(df[metric_col].sum()) if metric_col else "N/A"),
        (f"Total {secondary_metric_label}" if secondary_metric_col else "Records",
         fmt_num(df[secondary_metric_col].sum()) if secondary_metric_col else f"{len(df):,}"),
        (f"Active {category_label}s" if category_col else "Rows",
         f"{df[category_col].nunique():,}" if category_col else f"{len(df):,}"),
    ]
    try:
        pdf_bytes = generate_pdf_report(kpi_values, insights, ts, metric_label)
        st.download_button(
            "📄 Download Executive Summary (PDF)", data=pdf_bytes,
            file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.caption(f"PDF export unavailable right now ({e}).")

# ---------------------- TAB: ASK YOUR DATA ----------------------
with tab_ask:
    st.markdown("### Ask a question or give a task")
    st.caption(
        "Ask in plain language — e.g. \"Which group grew the most last month?\" The app "
        "writes and runs the calculation on your actual uploaded data, then explains the answer."
    )

    if not groq_api_key:
        st.warning(
            "Enter a free Groq API key in the sidebar under **🤖 Ask Your Data** to use this "
            "feature. Get one in about a minute at console.groq.com — no credit card needed."
        )
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant"):
                st.markdown(turn["answer"])
                if turn.get("result") is not None:
                    with st.expander("See the underlying data"):
                        if isinstance(turn["result"], (pd.DataFrame, pd.Series)):
                            st.dataframe(turn["result"], use_container_width=True)
                        else:
                            st.write(turn["result"])
                        st.code(turn["code"], language="python")

        question = st.chat_input("Ask a question about your data...")
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    outcome = answer_question(question, df, groq_api_key, groq_model)
                st.markdown(outcome["answer"])
                if outcome.get("result") is not None:
                    with st.expander("See the underlying data"):
                        if isinstance(outcome["result"], (pd.DataFrame, pd.Series)):
                            st.dataframe(outcome["result"], use_container_width=True)
                        else:
                            st.write(outcome["result"])
                        st.code(outcome["code"], language="python")
            st.session_state.chat_history.append({
                "question": question, "answer": outcome["answer"],
                "code": outcome["code"], "result": outcome["result"],
            })

        if st.session_state.chat_history:
            if st.button("Clear conversation"):
                st.session_state.chat_history = []
                st.rerun()

# ---------------------- TAB: CATEGORY PERFORMANCE ----------------------
if tab_category:
    with tab_category:
        if not metric_col:
            st.warning("No metric column confirmed — this tab needs one.")
        else:
            st.markdown(f"### Performance by {category_label}")
            col1, col2 = st.columns([2, 1])
            cat_df = df.groupby(category_col)[metric_col].sum().sort_values(ascending=False).reset_index()

            with col1:
                fig_bar = px.bar(cat_df, x=category_col, y=metric_col, title=f"{metric_label} by {category_label}",
                                  text_auto=".2s", color=metric_col, color_continuous_scale="Teal")
                fig_bar.update_layout(xaxis_title=category_label, yaxis_title=metric_label, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                fig_pie = px.pie(cat_df, names=category_col, values=metric_col, title="Share of Total", hole=0.45)
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("### Growth: Earlier Period vs. Later Period")
            growth_table = category_growth_table(df, category_col, metric_col, date_col)
            if growth_table is not None:
                display_table = growth_table.copy()
                display_table.columns = ["Earlier Period", "Later Period", "Change %"]
                st.dataframe(
                    display_table.style.format({
                        "Earlier Period": "{:,.0f}", "Later Period": "{:,.0f}", "Change %": "{:+.1f}%"
                    }).background_gradient(subset=["Change %"], cmap="RdYlGn"),
                    use_container_width=True,
                )
            else:
                st.info("Not enough date coverage to split into earlier/later periods.")

            if secondary_metric_col:
                st.markdown(f"### {metric_label} vs. {secondary_metric_label} by {category_label}")
                scatter_df = df.groupby(category_col)[[metric_col, secondary_metric_col]].sum().reset_index()
                fig_scatter = px.scatter(
                    scatter_df, x=metric_col, y=secondary_metric_col, text=category_col,
                    size=metric_col, color=category_col,
                    title=f"{metric_label} vs. {secondary_metric_label} (bubble = {metric_label.lower()} size)",
                )
                fig_scatter.update_traces(textposition="top center")
                st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------- TAB: SEGMENT ANALYSIS ----------------------
if tab_segment:
    with tab_segment:
        if not metric_col:
            st.warning("No metric column confirmed — segment analysis needs one.")
        else:
            st.markdown(f"### {segment_label} Performance")
            seg_totals = df.groupby(segment_col)[metric_col].agg(["sum", "mean", "count"]).reset_index()
            seg_totals.columns = [segment_col, "total", "average", "records"]
            seg_totals = seg_totals.sort_values("total", ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.bar(seg_totals, x=segment_col, y="total", title=f"Total {metric_label} by {segment_label}", text_auto=".2s")
                fig1.update_layout(xaxis_title=segment_label, yaxis_title=metric_label)
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.bar(seg_totals, x=segment_col, y="average", title=f"Average {metric_label} per Record",
                               text_auto=".2s", color_discrete_sequence=["#EF553B"])
                fig2.update_layout(xaxis_title=segment_label, yaxis_title=f"Avg {metric_label}")
                st.plotly_chart(fig2, use_container_width=True)

            if category_col:
                st.markdown(f"### {metric_label} Heatmap — {category_label} × {segment_label}")
                pivot = df.pivot_table(index=category_col, columns=segment_col, values=metric_col, aggfunc="sum", fill_value=0)
                fig_heat = px.imshow(pivot, text_auto=".2s", aspect="auto", color_continuous_scale="Blues")
                st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("### Summary Table")
            st.dataframe(seg_totals.rename(columns={"total": "Total", "average": "Average", "records": "Records"}), use_container_width=True)

# ---------------------- TAB: TRENDS & FORECAST ----------------------
if tab_trends:
    with tab_trends:
        if not metric_col:
            st.warning("No metric column confirmed — trend analysis needs one.")
        else:
            st.markdown("### Trend with Moving Average")
            ts, _ = period_over_period(df, metric_col, date_col, freq="W")
            if ts is not None and len(ts) >= 2:
                ma_window = min(4, len(ts))
                moving_avg = ts.rolling(window=ma_window, min_periods=1).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines+markers", name=metric_label))
                fig.add_trace(go.Scatter(x=moving_avg.index, y=moving_avg.values, mode="lines",
                                          name=f"{ma_window}-Period Moving Avg", line=dict(dash="dash")))
                fig.update_layout(title=f"{metric_label} — Weekly Trend & Moving Average", xaxis_title="Date", yaxis_title=metric_label)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Simple Trend Forecast (Next 4 Periods)")
                st.caption("A lightweight linear trend projection — useful for direction, not a substitute for a full forecasting model.")
                forecast, direction = linear_forecast(ts, periods_ahead=4)
                if forecast is not None:
                    fig_fc = go.Figure()
                    fig_fc.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines+markers", name="Actual"))
                    fig_fc.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers",
                                                 name="Forecast", line=dict(dash="dot", color="orange")))
                    fig_fc.update_layout(title=f"{metric_label} Forecast — Trend is {direction.title()}", xaxis_title="Date", yaxis_title=metric_label)
                    st.plotly_chart(fig_fc, use_container_width=True)
                    st.info(f"The underlying trend over the selected period is **{direction}**.")
                else:
                    st.info("Not enough weekly data points yet to build a trend forecast (need at least 4).")
            else:
                st.info("Not enough date coverage to plot a weekly trend.")

            if category_col:
                st.markdown(f"### Trend by {category_label}")
                multi_ts = df.dropna(subset=[date_col]).groupby(
                    [pd.Grouper(key=date_col, freq="W"), category_col]
                )[metric_col].sum().reset_index()
                fig_multi = px.line(multi_ts, x=date_col, y=metric_col, color=category_col,
                                     title=f"{metric_label} Over Time by {category_label}", markers=True)
                st.plotly_chart(fig_multi, use_container_width=True)

# ---------------------- TAB: SECONDARY METRIC ANALYSIS ----------------------
if tab_secondary:
    with tab_secondary:
        st.markdown(f"### {secondary_metric_label} Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Total {secondary_metric_label}", fmt_num(df[secondary_metric_col].sum()))
        with col2:
            if metric_col:
                total_secondary = df[secondary_metric_col].sum()
                ratio = df[metric_col].sum() / total_secondary * 100 if total_secondary else np.nan
                st.metric(f"Overall {metric_label} ÷ {secondary_metric_label} Ratio", f"{ratio:.1f}%" if not np.isnan(ratio) else "N/A")

        if category_col:
            st.markdown(f"### {secondary_metric_label} by {category_label}")
            sec_df = df.groupby(category_col)[secondary_metric_col].sum().sort_values(ascending=False).reset_index()
            fig_sec = px.bar(sec_df, x=category_col, y=secondary_metric_col, title=f"{secondary_metric_label} by {category_label}",
                              color=secondary_metric_col, color_continuous_scale="Purples")
            st.plotly_chart(fig_sec, use_container_width=True)

            if metric_col:
                st.markdown("### Ratio & Outliers")
                st.caption(
                    f"Ratio = {metric_label} ÷ {secondary_metric_label}. Unusually high values may mean "
                    f"{secondary_metric_label.lower()} isn't keeping pace; unusually low values may mean a surplus."
                )
                risk_df = df.groupby(category_col)[[metric_col, secondary_metric_col]].sum()
                risk_df["ratio_pct"] = np.where(risk_df[secondary_metric_col] > 0,
                                                 risk_df[metric_col] / risk_df[secondary_metric_col] * 100, np.nan)
                risk_df = risk_df.sort_values("ratio_pct", ascending=False)

                def flag(pct):
                    if pd.isna(pct):
                        return "No data"
                    if pct > 90:
                        return "⚠️ High"
                    if pct < 50:
                        return "📦 Low"
                    return "✅ Balanced"

                risk_df["status"] = risk_df["ratio_pct"].apply(flag)
                display_risk = risk_df.reset_index().rename(columns={
                    metric_col: metric_label, secondary_metric_col: secondary_metric_label,
                    "ratio_pct": "Ratio %", "status": "Status",
                })
                st.dataframe(
                    display_risk.style.format({metric_label: "{:,.0f}", secondary_metric_label: "{:,.0f}", "Ratio %": "{:.1f}%"}),
                    use_container_width=True,
                )

# ---------------------- TAB: DATA EXPLORER ----------------------
with tab_explorer:
    st.markdown("### Filtered, Cleaned Dataset")
    st.caption(f"{len(df):,} rows shown after sidebar filters are applied.")
    search_term = st.text_input("Search (matches any column, case-insensitive)")
    display_df = df.copy()
    if search_term:
        mask = display_df.astype(str).apply(lambda col: col.str.contains(search_term, case=False, na=False)).any(axis=1)
        display_df = display_df[mask]
    st.dataframe(display_df, use_container_width=True)

    st.markdown("### Download")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Filtered, Cleaned Dataset (CSV)", data=csv_buffer.getvalue(),
        file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv",
    )
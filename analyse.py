"""
FMCG Commercial Analytics Dashboard — Senior Analyst Edition
--------------------------------------------------------------
A drag-and-drop Streamlit app that turns a raw commercial/campaign/stock
export into a full analyst-style dashboard: cleaned data, KPIs, written
insights (auto-generated from the numbers, not hardcoded), account and
campaign deep-dives, a trend forecast, and stock risk flags.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    Push app.py + requirements.txt to a GitHub repo, then create the app
    at https://share.streamlit.io pointing at app.py.
"""

import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="FMCG Commercial Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 FMCG Commercial Analytics Dashboard")
st.caption(
    "Drop a CSV or Excel export below. The app cleans it, builds a full "
    "dashboard, and writes up the key findings the way a senior analyst would."
)

# ----------------------------------------------------------------------
# COLUMN ALIASES — maps messy source column names to canonical names
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
# CLEANING PIPELINE (same robust approach as before)
# ----------------------------------------------------------------------
def standardize_column_name(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^\w]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")


def map_to_canonical(columns):
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
    """Format large numbers with K/M suffixes for compact display."""
    if pd.isna(x):
        return "N/A"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000:
        return f"{sign}{prefix}{x/1_000_000:.2f}M{suffix}"
    if x >= 1_000:
        return f"{sign}{prefix}{x/1_000:.1f}K{suffix}"
    return f"{sign}{prefix}{x:,.0f}{suffix}"


# ----------------------------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📁 Drop your CSV or Excel file here",
    type=["csv", "xlsx"],
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.info("⬆️ Waiting for a file... Upload a commercial, campaign, or stock export.")
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
# IDENTIFY KEY COLUMNS
# ----------------------------------------------------------------------
volume_col = find_first_match(df_full, "volume", ["volume", "unit", "qty", "quantity", "case"])
revenue_col = find_first_match(df_full, "revenue", ["revenue", "sales", "turnover", "value", "amount"])
account_col = find_first_match(df_full, "account_name", ["account", "retail", "customer", "store", "outlet", "partner"])
date_col = find_first_match(df_full, "date", ["date", "period", "month", "week"])
stock_col = find_first_match(df_full, "stock", ["stock", "inventory", "allocat"])
campaign_col = find_first_match(df_full, "campaign", ["campaign", "promo"])

metric_col = revenue_col if revenue_col else volume_col
metric_label = (metric_col or "metric").replace("_", " ").title()

# ----------------------------------------------------------------------
# SIDEBAR FILTERS (apply globally to every tab)
# ----------------------------------------------------------------------
st.sidebar.header("🔎 Filters")
df = df_full.copy()

if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]) and df[date_col].notna().any():
    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()
    date_range = st.sidebar.date_input(
        "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        df = df[(df[date_col].dt.date >= start) & (df[date_col].dt.date <= end)]

if account_col:
    all_accounts = sorted(df_full[account_col].unique().tolist())
    selected_accounts = st.sidebar.multiselect("Retail accounts", all_accounts, default=all_accounts)
    df = df[df[account_col].isin(selected_accounts)]

if campaign_col:
    all_campaigns = sorted(df_full[campaign_col].unique().tolist())
    selected_campaigns = st.sidebar.multiselect("Campaigns", all_campaigns, default=all_campaigns)
    df = df[df[campaign_col].isin(selected_campaigns)]

if df.empty:
    st.warning("No rows match the current filters — widen your selection in the sidebar.")
    st.stop()

st.sidebar.caption(f"Showing {len(df):,} of {len(df_full):,} rows after filters.")

# ----------------------------------------------------------------------
# ANALYSIS HELPERS
# ----------------------------------------------------------------------
def period_over_period(df, metric_col, date_col, freq="W"):
    """Resample metric by freq and return the series + latest pct change."""
    if not (date_col and metric_col and pd.api.types.is_datetime64_any_dtype(df[date_col])):
        return None, None
    ts = df.dropna(subset=[date_col]).set_index(date_col)[metric_col].resample(freq).sum().sort_index()
    ts = ts[ts.index.notna()]
    if len(ts) < 2 or ts.iloc[-2] == 0:
        return ts, None
    pct = (ts.iloc[-1] - ts.iloc[-2]) / ts.iloc[-2] * 100
    return ts, pct


def account_growth_table(df, account_col, metric_col, date_col):
    """Split the filtered date range at the median date; compute % change
    per account between the earlier and later half."""
    if not (account_col and metric_col and date_col and pd.api.types.is_datetime64_any_dtype(df[date_col])):
        return None
    valid = df.dropna(subset=[date_col])
    if valid[date_col].nunique() < 2:
        return None
    midpoint = valid[date_col].median()
    first_half = valid[valid[date_col] <= midpoint].groupby(account_col)[metric_col].sum()
    second_half = valid[valid[date_col] > midpoint].groupby(account_col)[metric_col].sum()
    table = pd.DataFrame({"earlier_period": first_half, "later_period": second_half}).fillna(0)
    table["change_pct"] = np.where(
        table["earlier_period"] != 0,
        (table["later_period"] - table["earlier_period"]) / table["earlier_period"] * 100,
        np.nan,
    )
    return table.sort_values("later_period", ascending=False)


def linear_forecast(ts: pd.Series, periods_ahead: int = 4):
    """Fit a simple linear trend to a time series and project forward.
    This is a lightweight trend projection, not a statistical model —
    good for direction and rough magnitude, not for precise planning."""
    if ts is None or len(ts) < 4:
        return None, None
    y = ts.values.astype(float)
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)
    trend = np.poly1d(coeffs)
    future_x = np.arange(len(y), len(y) + periods_ahead)
    future_y = trend(future_x)
    future_y = np.clip(future_y, 0, None)
    freq = ts.index.freq or pd.infer_freq(ts.index)
    future_index = pd.date_range(ts.index[-1], periods=periods_ahead + 1, freq=freq or "W")[1:]
    forecast = pd.Series(future_y, index=future_index)
    slope_direction = "upward" if coeffs[0] > 0 else "downward" if coeffs[0] < 0 else "flat"
    return forecast, slope_direction


def generate_executive_insights(df, account_col, metric_col, metric_label, date_col, campaign_col, stock_col):
    """Build a list of plain-language, data-grounded findings — the kind
    of bullet points a senior analyst would open a report with."""
    insights = []

    if account_col and metric_col:
        by_account = df.groupby(account_col)[metric_col].sum().sort_values(ascending=False)
        total = by_account.sum()
        if len(by_account) > 0 and total > 0:
            top_name, top_val = by_account.index[0], by_account.iloc[0]
            top_share = top_val / total * 100
            insights.append(
                f"**{top_name}** is the leading account, contributing **{fmt_num(top_val)}** "
                f"in {metric_label.lower()} — **{top_share:.1f}%** of the total across all "
                f"{len(by_account)} accounts."
            )
            if len(by_account) > 1:
                bottom_name, bottom_val = by_account.index[-1], by_account.iloc[-1]
                insights.append(
                    f"**{bottom_name}** is the weakest performer at **{fmt_num(bottom_val)}**, "
                    f"roughly **{(top_val / bottom_val):.1f}x** below the top account — "
                    f"worth a closer look at execution or distribution there."
                )

    ts, pct = period_over_period(df, metric_col, date_col, freq="W")
    if pct is not None:
        direction = "grew" if pct >= 0 else "declined"
        insights.append(
            f"The most recent week-over-week {metric_label.lower()} trend **{direction} by "
            f"{abs(pct):.1f}%** versus the prior week."
        )

    growth_table = account_growth_table(df, account_col, metric_col, date_col)
    if growth_table is not None and not growth_table["change_pct"].isna().all():
        gt = growth_table.dropna(subset=["change_pct"])
        if not gt.empty:
            best = gt["change_pct"].idxmax()
            worst = gt["change_pct"].idxmin()
            insights.append(
                f"Comparing the first and second half of the selected period, **{best}** posted "
                f"the strongest growth (**{gt.loc[best, 'change_pct']:+.1f}%**), while **{worst}** "
                f"saw the largest decline (**{gt.loc[worst, 'change_pct']:+.1f}%**)."
            )

    if campaign_col and metric_col:
        by_campaign = df.groupby(campaign_col)[metric_col].mean().sort_values(ascending=False)
        if len(by_campaign) > 1:
            best_campaign = by_campaign.index[0]
            insights.append(
                f"**{best_campaign}** shows the highest average {metric_label.lower()} per record "
                f"among tracked campaigns — a candidate to prioritize or extend."
            )

    if stock_col and volume_col and pd.api.types.is_numeric_dtype(df[stock_col]):
        sell_through = df.groupby(account_col)[[volume_col, stock_col]].sum() if account_col else None
        if sell_through is not None and (sell_through[stock_col] > 0).any():
            sell_through["sell_through_rate"] = sell_through[volume_col] / sell_through[stock_col].replace(0, np.nan)
            at_risk = sell_through[sell_through["sell_through_rate"] > 0.9]
            if not at_risk.empty:
                names = ", ".join(at_risk.index.tolist()[:3])
                insights.append(
                    f"⚠️ **Stockout risk**: {names} show a sell-through rate above 90% — "
                    f"stock allocation may not be keeping pace with demand."
                )

    if not insights:
        insights.append(
            "Not enough recognizable commercial columns (account, volume/revenue, date) were "
            "found to generate detailed findings — check the data preview below."
        )

    return insights


# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tab_summary, tab_accounts, tab_campaigns, tab_trends, tab_stock, tab_explorer = st.tabs(
    ["📋 Executive Summary", "🏬 Account Performance", "🎯 Campaign Analysis",
     "📈 Trends & Forecast", "📦 Stock & Risk", "🔍 Data Explorer"]
)

# ---------------------- TAB 1: EXECUTIVE SUMMARY ----------------------
with tab_summary:
    st.markdown("### Key Performance Indicators")
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("Total Volume", fmt_num(df[volume_col].sum()) if volume_col else "N/A")
    with kpi_cols[1]:
        st.metric("Total Revenue", fmt_num(df[revenue_col].sum(), prefix="€") if revenue_col else "N/A")
    with kpi_cols[2]:
        st.metric("Active Retail Accounts", f"{df[account_col].nunique():,}" if account_col else "N/A")
    with kpi_cols[3]:
        _, pct = period_over_period(df, metric_col, date_col, freq="W")
        st.metric("Latest Week-over-Week Growth", f"{pct:+.1f}%" if pct is not None else "N/A")

    st.divider()
    st.markdown("### 🧠 Analyst Findings")
    st.caption("Auto-generated directly from the filtered data above — not templated text.")
    insights = generate_executive_insights(
        df, account_col, metric_col, metric_label, date_col, campaign_col, stock_col
    )
    for point in insights:
        st.markdown(f"- {point}")

    st.divider()
    st.markdown("### Overview")
    ts, _ = period_over_period(df, metric_col, date_col, freq="W")
    if ts is not None and len(ts) > 0:
        fig = px.area(
            x=ts.index, y=ts.values,
            title=f"{metric_label} Trend (Weekly)", labels={"x": "Date", "y": metric_label},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No date column detected — trend overview unavailable.")

# ---------------------- TAB 2: ACCOUNT PERFORMANCE ----------------------
with tab_accounts:
    if not (account_col and metric_col):
        st.warning("No account or volume/revenue column detected — this tab needs both.")
    else:
        st.markdown("### Performance by Retail Account")
        col1, col2 = st.columns([2, 1])

        acc_df = df.groupby(account_col)[metric_col].sum().sort_values(ascending=False).reset_index()
        with col1:
            fig_bar = px.bar(
                acc_df, x=account_col, y=metric_col,
                title=f"{metric_label} by Account", text_auto=".2s", color=metric_col,
                color_continuous_scale="Teal",
            )
            fig_bar.update_layout(xaxis_title="Account", yaxis_title=metric_label, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_pie = px.pie(
                acc_df, names=account_col, values=metric_col,
                title="Market Share", hole=0.45,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### Growth: Earlier Period vs. Later Period")
        growth_table = account_growth_table(df, account_col, metric_col, date_col)
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

        if volume_col and revenue_col and volume_col != revenue_col:
            st.markdown("### Volume vs. Revenue by Account")
            scatter_df = df.groupby(account_col)[[volume_col, revenue_col]].sum().reset_index()
            fig_scatter = px.scatter(
                scatter_df, x=volume_col, y=revenue_col, text=account_col,
                size=revenue_col, color=account_col, title="Volume vs. Revenue (bubble = revenue size)",
            )
            fig_scatter.update_traces(textposition="top center")
            st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------- TAB 3: CAMPAIGN ANALYSIS ----------------------
with tab_campaigns:
    if not campaign_col:
        st.info(
            "No campaign column was detected in this file. Add a column like "
            "`campaign_name` or `promo` to unlock this analysis."
        )
    elif not metric_col:
        st.warning("No volume/revenue column detected — campaign analysis needs a metric to compare.")
    else:
        st.markdown("### Campaign Performance")
        camp_totals = df.groupby(campaign_col)[metric_col].agg(["sum", "mean", "count"]).reset_index()
        camp_totals.columns = [campaign_col, "total", "average", "records"]
        camp_totals = camp_totals.sort_values("total", ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.bar(
                camp_totals, x=campaign_col, y="total",
                title=f"Total {metric_label} by Campaign", text_auto=".2s",
            )
            fig1.update_layout(xaxis_title="Campaign", yaxis_title=metric_label)
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.bar(
                camp_totals, x=campaign_col, y="average",
                title=f"Average {metric_label} per Record", text_auto=".2s", color_discrete_sequence=["#EF553B"],
            )
            fig2.update_layout(xaxis_title="Campaign", yaxis_title=f"Avg {metric_label}")
            st.plotly_chart(fig2, use_container_width=True)

        if account_col:
            st.markdown("### Campaign Performance by Account")
            pivot = df.pivot_table(index=account_col, columns=campaign_col, values=metric_col, aggfunc="sum", fill_value=0)
            fig_heat = px.imshow(
                pivot, text_auto=".2s", aspect="auto",
                title=f"{metric_label} Heatmap — Account × Campaign", color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("### Campaign Summary Table")
        st.dataframe(
            camp_totals.rename(columns={"total": "Total", "average": "Average", "records": "Records"}),
            use_container_width=True,
        )

# ---------------------- TAB 4: TRENDS & FORECAST ----------------------
with tab_trends:
    if not (date_col and metric_col and pd.api.types.is_datetime64_any_dtype(df[date_col])):
        st.warning("No usable date column detected — trend and forecast analysis needs one.")
    else:
        st.markdown("### Trend with Moving Average")
        ts, _ = period_over_period(df, metric_col, date_col, freq="W")
        if ts is not None and len(ts) >= 2:
            ma_window = min(4, len(ts))
            moving_avg = ts.rolling(window=ma_window, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines+markers", name=metric_label))
            fig.add_trace(go.Scatter(x=moving_avg.index, y=moving_avg.values, mode="lines",
                                      name=f"{ma_window}-Week Moving Avg", line=dict(dash="dash")))
            fig.update_layout(title=f"{metric_label} — Weekly Trend & Moving Average",
                               xaxis_title="Date", yaxis_title=metric_label)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Simple Trend Forecast (Next 4 Periods)")
            st.caption(
                "This is a lightweight linear trend projection based on recent history — useful "
                "for direction and rough magnitude, not a substitute for a full forecasting model."
            )
            forecast, direction = linear_forecast(ts, periods_ahead=4)
            if forecast is not None:
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines+markers", name="Actual"))
                fig_fc.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers",
                                             name="Forecast", line=dict(dash="dot", color="orange")))
                fig_fc.update_layout(title=f"{metric_label} Forecast — Trend is {direction.title()}",
                                      xaxis_title="Date", yaxis_title=metric_label)
                st.plotly_chart(fig_fc, use_container_width=True)
                st.info(f"The underlying trend over the selected period is **{direction}**.")
            else:
                st.info("Not enough weekly data points yet to build a trend forecast (need at least 4).")
        else:
            st.info("Not enough date coverage to plot a weekly trend.")

        if account_col:
            st.markdown("### Trend by Account")
            multi_ts = df.dropna(subset=[date_col]).groupby(
                [pd.Grouper(key=date_col, freq="W"), account_col]
            )[metric_col].sum().reset_index()
            fig_multi = px.line(
                multi_ts, x=date_col, y=metric_col, color=account_col,
                title=f"{metric_label} Over Time by Account", markers=True,
            )
            st.plotly_chart(fig_multi, use_container_width=True)

# ---------------------- TAB 5: STOCK & RISK ----------------------
with tab_stock:
    if not stock_col:
        st.info(
            "No stock/inventory column was detected in this file. Add a column like "
            "`stock_allocation` or `inventory` to unlock this analysis."
        )
    else:
        st.markdown("### Stock Allocation Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Stock Allocated", fmt_num(df[stock_col].sum()))
        with col2:
            if volume_col:
                overall_sell_through = df[volume_col].sum() / df[stock_col].sum() * 100 if df[stock_col].sum() else np.nan
                st.metric("Overall Sell-Through Rate", f"{overall_sell_through:.1f}%" if not np.isnan(overall_sell_through) else "N/A")

        if account_col:
            st.markdown("### Stock Allocation by Account")
            stock_df = df.groupby(account_col)[stock_col].sum().sort_values(ascending=False).reset_index()
            fig_stock = px.bar(
                stock_df, x=account_col, y=stock_col, title="Stock Allocation by Account",
                color=stock_col, color_continuous_scale="Purples",
            )
            st.plotly_chart(fig_stock, use_container_width=True)

            if volume_col:
                st.markdown("### Sell-Through Rate & Stockout Risk")
                st.caption(
                    "Sell-through = volume sold ÷ stock allocated. Rates above 90% suggest demand "
                    "may be outpacing supply; rates well below 50% suggest potential overstock."
                )
                risk_df = df.groupby(account_col)[[volume_col, stock_col]].sum()
                risk_df["sell_through_pct"] = np.where(
                    risk_df[stock_col] > 0, risk_df[volume_col] / risk_df[stock_col] * 100, np.nan
                )
                risk_df = risk_df.sort_values("sell_through_pct", ascending=False)

                def flag_risk(pct):
                    if pd.isna(pct):
                        return "No data"
                    if pct > 90:
                        return "⚠️ Stockout risk"
                    if pct < 50:
                        return "📦 Possible overstock"
                    return "✅ Healthy"

                risk_df["status"] = risk_df["sell_through_pct"].apply(flag_risk)
                display_risk = risk_df.reset_index().rename(columns={
                    volume_col: "Volume", stock_col: "Stock", "sell_through_pct": "Sell-Through %",
                    "status": "Status",
                })
                st.dataframe(
                    display_risk.style.format({"Volume": "{:,.0f}", "Stock": "{:,.0f}", "Sell-Through %": "{:.1f}%"}),
                    use_container_width=True,
                )

# ---------------------- TAB 6: DATA EXPLORER ----------------------
with tab_explorer:
    st.markdown("### Filtered, Cleaned Dataset")
    st.caption(f"{len(df):,} rows shown after sidebar filters are applied.")
    search_term = st.text_input("Search (matches any column, case-insensitive)")
    display_df = df.copy()
    if search_term:
        mask = display_df.astype(str).apply(
            lambda col: col.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]
    st.dataframe(display_df, use_container_width=True)

    st.markdown("### Download")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Filtered, Cleaned Dataset (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
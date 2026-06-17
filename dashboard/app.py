"""
Pharmaceutical Demand Forecasting Dashboard
Data Science / ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Pharma Demand Forecasting",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size:2rem; font-weight:700; color:#1a3c5e; margin-bottom:0}
    .sub-header  {font-size:1rem; color:#666; margin-bottom:1.5rem}
    .metric-card {background:#f0f4f8; border-radius:8px; padding:1rem; text-align:center}
    .stSelectbox label {font-weight:600}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df_raw = pd.read_csv("data/raw/synthetic_sales_data.csv", parse_dates=["date"])
    df_agg = pd.read_csv("data/raw/aggregated_monthly_sales.csv", parse_dates=["date"])
    return df_raw, df_agg

df_raw, df_agg = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Filters")

product = st.sidebar.selectbox("Select Product", sorted(df_agg["product_name"].unique()))
region  = st.sidebar.selectbox("Select Region", ["All Regions"] + sorted(df_raw["region"].unique().tolist()))
model   = st.sidebar.radio("Forecast Model", ["Prophet", "SARIMAX"], index=0)
horizon = st.sidebar.slider("Forecast Horizon (months)", 3, 12, 6)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Pharmaceutical Demand Forecasting")
st.sidebar.markdown("**Domain:** AI / ML | Time Series")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">💊 Pharmaceutical Demand Forecasting System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI/ML-powered sales prediction & inventory optimization system</p>', unsafe_allow_html=True)
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
prod_data = df_agg[df_agg["product_name"] == product]

total_units   = prod_data["total_units_sold"].sum()
avg_monthly   = prod_data["total_units_sold"].mean()
total_revenue = prod_data["total_revenue_inr"].sum()
yoy_growth    = ((prod_data[prod_data["year"] == 2024]["total_units_sold"].sum() /
                  prod_data[prod_data["year"] == 2023]["total_units_sold"].sum()) - 1) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("📦 Total Units (3yr)",  f"{total_units:,}")
col2.metric("📅 Avg Monthly Demand", f"{avg_monthly:,.0f}")
col3.metric("💰 Total Revenue",      f"₹{total_revenue:,.0f}")
col4.metric("📈 YoY Growth (23→24)", f"{yoy_growth:+.1f}%")

st.markdown("---")

# ── Historical Trend ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"📊 Historical Sales — {product}")
    if region == "All Regions":
        plot_data = prod_data.sort_values("date")
        y_col = "total_units_sold"
    else:
        plot_data = df_raw[(df_raw["product_name"] == product) & (df_raw["region"] == region)].sort_values("date")
        y_col = "units_sold"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_data["date"], plot_data[y_col], color="steelblue", linewidth=2, marker="o", markersize=3)
    ax.fill_between(plot_data["date"], plot_data[y_col], alpha=0.1, color="steelblue")
    ax.set_ylabel("Units Sold")
    ax.set_title(f"{product} — {'All Regions' if region == 'All Regions' else region}", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("🏆 Top Products by Revenue")
    top_rev = df_agg.groupby("product_name")["total_revenue_inr"].sum().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    colors = ["#1a3c5e" if p == product else "#90aec4" for p in top_rev.index]
    ax2.barh(top_rev.index, top_rev.values, color=colors)
    ax2.set_xlabel("Revenue (INR)")
    ax2.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ── Forecast Section ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"🔮 Demand Forecast — Next {horizon} Months ({model})")

series = prod_data.set_index("date")["total_units_sold"].asfreq("MS")

if model == "Prophet":
    try:
        from prophet import Prophet
        train_df = prod_data[["date", "total_units_sold"]].rename(columns={"date": "ds", "total_units_sold": "y"})

        holidays = pd.DataFrame({
            'holiday': ['Diwali','Diwali','Diwali','Holi','Holi','Holi'],
            'ds': pd.to_datetime([
                '2022-10-24','2023-11-12','2024-11-01',
                '2022-03-18','2023-03-08','2024-03-25',
            ]),
            'lower_window': [-2,-2,-2,-1,-1,-1],
            'upper_window': [1,1,1,1,1,1],
        })

        m = Prophet(
            seasonality_mode='additive',
            yearly_seasonality=8,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays,
            changepoint_prior_scale=0.15,
            seasonality_prior_scale=5,
        )
        m.fit(train_df)
        future = m.make_future_dataframe(periods=horizon, freq="MS")
        fc = m.predict(future)
        forecast_vals = fc.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        forecast_vals.columns = ["Date", "Forecast", "Lower", "Upper"]
        forecast_vals[["Forecast", "Lower", "Upper"]] = forecast_vals[["Forecast", "Lower", "Upper"]].clip(lower=0).round(0).astype(int)

        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.plot(fc["ds"], fc["yhat"], color="tomato", linewidth=2, linestyle="--", label="Forecast")
        ax3.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], alpha=0.15, color="tomato")
        ax3.plot(train_df["ds"], train_df["y"], color="steelblue", linewidth=2, label="Historical")
        ax3.axvline(train_df["ds"].max(), color="gray", linestyle=":", linewidth=1.5)
        ax3.set_title(f"Prophet Forecast — {product}", fontweight="bold")
        ax3.legend()
        ax3.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        st.subheader("📋 Forecast Table")
        st.dataframe(forecast_vals.reset_index(drop=True), use_container_width=True)

        csv = forecast_vals.to_csv(index=False)
        st.download_button("⬇️ Download Forecast CSV", csv, f"{product}_prophet_forecast.csv", "text/csv")

    except ImportError:
        st.error("Prophet not installed. Run: pip install prophet")

else:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model_fit = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,0,12),
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = model_fit.get_forecast(steps=horizon)
        pred = fc.predicted_mean
        ci   = fc.conf_int(alpha=0.2)

        fig4, ax4 = plt.subplots(figsize=(12, 5))
        ax4.plot(series.index, series.values, color="steelblue", linewidth=2, label="Historical")
        ax4.plot(pred.index, pred.values, color="tomato", linewidth=2, linestyle="--", label="SARIMAX Forecast", marker="o")
        ax4.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.15, color="tomato")
        ax4.axvline(series.index[-1], color="gray", linestyle=":", linewidth=1.5)
        ax4.set_title(f"SARIMAX Forecast — {product}", fontweight="bold")
        ax4.legend()
        ax4.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

        forecast_table = pd.DataFrame({
            "Date": pred.index.strftime("%Y-%m"),
            "Forecast": pred.values.round(0).astype(int),
            "Lower (80% CI)": ci.iloc[:,0].round(0).astype(int),
            "Upper (80% CI)": ci.iloc[:,1].round(0).astype(int),
        })
        st.subheader("📋 Forecast Table")
        st.dataframe(forecast_table, use_container_width=True)

        csv = forecast_table.to_csv(index=False)
        st.download_button("⬇️ Download Forecast CSV", csv, f"{product}_sarimax_forecast.csv", "text/csv")

    except ImportError:
        st.error("statsmodels not installed. Run: pip install statsmodels")

# ── Seasonality Heatmap ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🗓️ Seasonality Heatmap")

pivot = prod_data.pivot_table(index="year", columns="month_name", values="total_units_sold", aggfunc="sum")
month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
pivot = pivot[[m for m in month_order if m in pivot.columns]]

fig5, ax5 = plt.subplots(figsize=(12, 3))
import seaborn as sns
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax5, linewidths=0.5)
ax5.set_title(f"Monthly Sales Heatmap — {product}", fontweight="bold")
plt.tight_layout()
st.pyplot(fig5)
plt.close()

st.markdown("---")
st.caption("Pharmaceutical Demand Forecasting System | Data Science Project ")
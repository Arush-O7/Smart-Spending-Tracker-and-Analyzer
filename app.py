import io
import re
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from datetime import datetime, timezone
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# Try to import scikit-learn for advanced anomaly detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ---------------- Config ----------------
MODEL_PATH = "models/categorization_pipeline.joblib"
RISK_MODEL_PATH = "models/risk_engine_model.joblib"
AMOUNT_BINS = None

# Canonical columns expected downstream
REQUIRED_CANON = ["timestamp", "merchant", "amount"]
OPTIONAL_CANON = ["city", "state", "category"]

# Alias map (normalized: lowercase, remove non-alphanum)
ALIASES = {
    "timestamp": {
        "transdatetranstime","transactiontime","transactiondatetime",
        "datetime","date","time","timestamp","transtime","transdatetime",
        "transdate","transtimestamp"
    },
    "merchant": {
        "merchant","merchantname","vendor","payee","description","narration",
        "merchantdesc","merchantdescription","merchanttext","merchant_details",
        "merchant_name"
    },
    "amount": {
        "amount","amt","amnt","transactionamount","txnamount","value",
        "debit","credit","amountinr","amtinr","totalamount","amount_rs",
        "amountinrs","txn_amt","amountrs"
    },
    "city": {"city","merchantcity","billingcity","txncity"},
    "state": {"state","merchantstate","region","province","txnstate"},
    "category": {"category","label","class","txncategory","merchantcategory"},
}

# ---------------- Utilities: schema detection ----------------
def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())

def guess_schema_columns(df: pd.DataFrame):
    norm_map = {_normalize_col(c): c for c in df.columns}
    found = {}

    for canon, alias_set in ALIASES.items():
        match = None
        for norm, orig in norm_map.items():
            if norm == canon or norm in alias_set:
                match = orig
                break
        found[canon] = match

    if found["amount"] is None:
        for norm, orig in norm_map.items():
            if "amount" in norm:
                found["amount"] = orig
                break
    if found["timestamp"] is None:
        for norm, orig in norm_map.items():
            if "date" in norm or "time" in norm or norm.endswith("ts") or "timestamp" in norm:
                found["timestamp"] = orig
                break
    if found["merchant"] is None:
        for norm, orig in norm_map.items():
            if "desc" in norm or "narrat" in norm or "vendor" in norm or "payee" in norm or "merchant" in norm:
                found["merchant"] = orig
                break

    missing_req = [c for c in REQUIRED_CANON if found.get(c) is None]
    return found, missing_req

def align_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    mapping, missing = guess_schema_columns(df)
    if missing:
        raise ValueError(
            "Missing required columns after auto-detection: "
            f"{missing}. Rename headers or provide recognizable aliases."
        )
    rename_map = {
        mapping["timestamp"]: "timestamp",
        mapping["merchant"]: "merchant",
        mapping["amount"]: "amount",
    }
    for opt in OPTIONAL_CANON:
        if mapping.get(opt):
            rename_map[mapping[opt]] = opt

    df_out = df.rename(columns=rename_map)
    keep = [c for c in ["timestamp","merchant","amount","city","state","category"] if c in df_out.columns]
    return df_out[keep] if keep else df_out

# ---------------- Feature engineering ----------------
def build_features(df: pd.DataFrame):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["merchant"] = (
        df["merchant"].fillna("")
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    before = len(df)
    df = df.dropna(subset=["timestamp","amount"]).reset_index(drop=True)
    dropped = before - len(df)

    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour
    df["log_amount"] = np.log1p(df["amount"])

    if AMOUNT_BINS:
        df["amount_bin"] = pd.cut(df["amount"], bins=AMOUNT_BINS, labels=False, include_lowest=True)
    else:
        try:
            df["amount_bin"] = pd.qcut(df["amount"], q=4, labels=False, duplicates="drop")
        except Exception:
            df["amount_bin"] = pd.cut(
                df["amount"],
                bins=[-np.inf, 10, 50, 200, np.inf],
                labels=False,
                include_lowest=True,
            )
    return df, dropped

# ---------------- NEW: Advanced Anomaly Detection ----------------
def detect_multi_dimensional_anomalies(facts: pd.DataFrame, contamination=0.05):
    """
    Multi-dimensional anomaly detection using Isolation Forest on engineered features
    """
    if not SKLEARN_AVAILABLE or facts.empty or len(facts) < 10:
        return pd.DataFrame()
    
    df = facts.copy()
    
    # Create feature matrix
    feature_cols = []
    features = pd.DataFrame(index=df.index)
    
    # Temporal features
    features['hour'] = df['timestamp'].dt.hour
    features['day_of_week'] = df['timestamp'].dt.dayofweek
    features['day_of_month'] = df['timestamp'].dt.day
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
    
    # Amount features
    features['amount'] = df['amount']
    features['log_amount'] = np.log1p(df['amount'])
    
    # Merchant frequency features
    merchant_counts = df['merchant'].value_counts()
    features['merchant_freq'] = df['merchant'].map(merchant_counts)
    features['is_rare_merchant'] = (features['merchant_freq'] <= 3).astype(int)
    
    # Category features if available
    if 'pred_category' in df.columns:
        cat_amounts = df.groupby('pred_category')['amount'].transform('median')
        features['amount_vs_category_median'] = df['amount'] / (cat_amounts + 1)
    
    # Time-based features (transactions per day)
    df['date'] = df['timestamp'].dt.date
    daily_counts = df.groupby('date').size()
    features['daily_txn_count'] = df['date'].map(daily_counts)
    
    # Prepare for Isolation Forest
    X = features.fillna(0).values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.score_samples(X_scaled)
    
    # Create results dataframe
    df['anomaly_score'] = -anomaly_scores  # Negate so higher = more anomalous
    df['is_anomaly'] = (anomaly_labels == -1)
    
    # Add interpretable reasons
    reasons = []
    for idx, row in df.iterrows():
        reason_list = []
        if features.loc[idx, 'is_rare_merchant']:
            reason_list.append("rare_merchant")
        if features.loc[idx, 'amount'] > df['amount'].quantile(0.95):
            reason_list.append("high_amount")
        if features.loc[idx, 'daily_txn_count'] > df.groupby('date').size().quantile(0.90):
            reason_list.append("high_frequency_day")
        if features.loc[idx, 'hour'] < 6 or features.loc[idx, 'hour'] > 23:
            reason_list.append("unusual_hour")
        reasons.append(",".join(reason_list) if reason_list else "pattern_based")
    
    df['anomaly_reason'] = reasons
    
    anomalies = df[df['is_anomaly']].sort_values('anomaly_score', ascending=False)
    display_cols = ['timestamp', 'merchant', 'amount', 'anomaly_score', 'anomaly_reason']
    if 'pred_category' in anomalies.columns:
        display_cols.insert(3, 'pred_category')
    
    return anomalies[display_cols].head(50)

# ---------------- NEW: Spending Pattern Analysis ----------------
def analyze_spending_patterns(facts: pd.DataFrame):
    """
    Identify unique spending patterns and behavioral insights
    """
    if facts.empty:
        return {}
    
    patterns = {}
    
    # 1. Weekday vs Weekend spending
    facts['is_weekend'] = facts['timestamp'].dt.dayofweek >= 5
    weekend_avg = facts[facts['is_weekend']]['amount'].mean()
    weekday_avg = facts[~facts['is_weekend']]['amount'].mean()
    patterns['weekend_vs_weekday_ratio'] = weekend_avg / (weekday_avg + 0.01)
    
    # 2. Time-of-day patterns
    facts['hour'] = facts['timestamp'].dt.hour
    facts['time_period'] = pd.cut(facts['hour'], 
                                   bins=[0, 6, 12, 18, 24],
                                   labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                   include_lowest=True)
    patterns['time_distribution'] = facts.groupby('time_period')['amount'].agg(['sum', 'count']).to_dict()
    
    # 3. Spending velocity (trend over time)
    facts_sorted = facts.sort_values('timestamp')
    facts_sorted['week_num'] = (facts_sorted['timestamp'] - facts_sorted['timestamp'].min()).dt.days // 7
    weekly_spend = facts_sorted.groupby('week_num')['amount'].sum()
    if len(weekly_spend) > 2:
        patterns['spending_trend'] = 'increasing' if weekly_spend.iloc[-3:].mean() > weekly_spend.iloc[:3].mean() else 'decreasing'
    
    # 4. Merchant diversity (concentration)
    merchant_entropy = -((facts['merchant'].value_counts(normalize=True) * 
                         np.log(facts['merchant'].value_counts(normalize=True))).sum())
    patterns['merchant_diversity_score'] = float(merchant_entropy)
    
    # 5. Average transaction value by day of week
    patterns['dow_avg_amount'] = facts.groupby(facts['timestamp'].dt.day_name())['amount'].mean().to_dict()
    
    return patterns

# ---------------- NEW: Predictive Insights ----------------
def generate_predictive_insights(facts: pd.DataFrame):
    """
    Generate forward-looking insights based on historical patterns
    """
    if facts.empty or len(facts) < 30:
        return {}
    
    insights = {}
    
    # Project next month spending based on trend
    facts_sorted = facts.sort_values('timestamp')
    facts_sorted['month'] = facts_sorted['timestamp'].dt.to_period('M')
    monthly_spend = facts_sorted.groupby('month')['amount'].sum()
    
    if len(monthly_spend) >= 3:
        # Simple linear projection
        recent_avg = monthly_spend.iloc[-3:].mean()
        insights['projected_next_month'] = float(recent_avg)
        
        # Identify categories likely to increase
        if 'pred_category' in facts.columns:
            cat_monthly = facts_sorted.groupby(['month', 'pred_category'])['amount'].sum().reset_index()
            growing_cats = []
            for cat in facts['pred_category'].unique():
                cat_data = cat_monthly[cat_monthly['pred_category'] == cat]['amount']
                if len(cat_data) >= 3:
                    if cat_data.iloc[-1] > cat_data.iloc[-3]:
                        growing_cats.append(cat)
            insights['growing_categories'] = growing_cats
    
    # Identify merchants you might revisit soon
    merchant_revisit = []
    for merchant in facts['merchant'].value_counts().head(10).index:
        merchant_txns = facts[facts['merchant'] == merchant].sort_values('timestamp')
        if len(merchant_txns) >= 3:
            intervals = merchant_txns['timestamp'].diff().dt.days.dropna()
            avg_interval = intervals.mean()
            last_txn = merchant_txns['timestamp'].max()
            days_since = (pd.Timestamp.now() - last_txn).days
            if days_since >= avg_interval * 0.8:
                merchant_revisit.append({
                    'merchant': merchant,
                    'avg_interval_days': float(avg_interval),
                    'days_since_last': int(days_since)
                })
    insights['merchants_due_for_revisit'] = merchant_revisit[:5]
    
    return insights

# ---------------- Insights helpers ----------------
def make_facts(out_df: pd.DataFrame) -> pd.DataFrame:
    facts = out_df.copy()
    facts["timestamp"] = pd.to_datetime(facts["timestamp"], errors="coerce")
    facts = facts.dropna(subset=["timestamp","amount"]).reset_index(drop=True)
    facts["date"] = facts["timestamp"].dt.date
    facts["week"] = facts["timestamp"].dt.to_period("W").astype(str)
    facts["month"] = facts["timestamp"].dt.to_period("M").astype(str)
    return facts

def recurring_detection(facts: pd.DataFrame):
    if facts.empty:
        return pd.DataFrame(columns=["merchant","count","median_interval_days","amount_cv","last_seen","next_estimated"])
    df = facts.sort_values("timestamp").copy()
    g = df.groupby("merchant", as_index=False)
    intervals = (
        df.groupby("merchant")["timestamp"]
        .apply(lambda s: s.sort_values().diff().dropna().dt.days)
    )
    if intervals.empty:
        return pd.DataFrame(columns=["merchant","count","median_interval_days","amount_cv","last_seen","next_estimated"])
    med_interval = intervals.groupby(level=0).median().rename("median_interval_days")
    amt_stats = g["amount"].agg(["mean","std","count"]).rename(columns={"mean":"mu","std":"sigma"})
    amt_stats["amount_cv"] = amt_stats.apply(lambda r: (r["sigma"] / r["mu"]) if r["mu"] and r["sigma"] is not None else np.nan, axis=1)
    last_seen = g["timestamp"].max().rename(columns={"timestamp":"last_seen"})
    rec = amt_stats.join(med_interval, how="left").merge(last_seen, on="merchant", how="left")
    rec = rec.rename(columns={"count":"count"})
    rec["median_interval_days"] = rec["median_interval_days"].fillna(np.inf)
    mask = (
        (rec["count"] >= 3) &
        (rec["median_interval_days"].between(26, 33)) &
        (rec["amount_cv"].fillna(1.0) < 0.2)
    )
    rec = rec.loc[mask, ["merchant","count","median_interval_days","amount_cv","last_seen"]].copy()
    if not rec.empty:
        rec["next_estimated"] = rec["last_seen"] + rec["median_interval_days"].apply(lambda d: pd.Timedelta(days=float(d)))
    return rec.sort_values(["median_interval_days","merchant"]).reset_index(drop=True)

def anomaly_high_spend(facts: pd.DataFrame, top_n=5):
    if facts.empty:
        return facts
    by_cat = facts.groupby("pred_category")["amount"]
    mu = by_cat.transform("mean")
    sigma = by_cat.transform("std").replace(0, np.nan)
    facts = facts.copy()
    facts["z"] = (facts["amount"] - mu) / sigma
    out = facts.dropna(subset=["z"]).sort_values("z", ascending=False)
    return out[["timestamp","merchant","amount","pred_category","z"]].head(top_n)

# ---------------- Risk feature building ----------------
def build_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    use_cols = {
        "trans_date_trans_time": "timestamp",
        "merchant": "merchant",
        "category": "category",
        "amt": "amount",
        "city": "city",
        "state": "state",
        "cc_num": "cc_num",
        "unix_time": "unix_time",
        "lat": "cust_lat",
        "long": "cust_long",
        "merch_lat": "merch_lat",
        "merch_long": "merch_long",
    }
    keep = [c for c in use_cols if c in df.columns]
    x = df[keep].rename(columns=use_cols).copy()

    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    for c in ["amount","unix_time","cust_lat","cust_long","merch_lat","merch_long"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    for c in ["merchant","city","state"]:
        if c in x.columns:
            x[c] = (x[c].fillna("").str.lower()
                    .str.replace(r"[^\w\s]", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True).str.strip())
    x = x.dropna(subset=["timestamp","amount"]).reset_index(drop=True)

    has_card = ("cc_num" in x.columns) and x["cc_num"].notna().any()
    sort_keys = ["cc_num","timestamp"] if has_card else ["timestamp"]
    x = x.sort_values(sort_keys)
    group_keys = ["cc_num"] if has_card else None

    def add_roll(g, N=20):
        g = g.sort_values("timestamp").copy()
        g["prev_txn_delta_min"] = g["timestamp"].diff().dt.total_seconds().div(60).fillna(1e6)
        g["roll_cnt_N"]  = g["amount"].rolling(N, min_periods=1).count()
        g["roll_sum_N"]  = g["amount"].rolling(N, min_periods=1).sum()
        g["roll_max_N"]  = g["amount"].rolling(N, min_periods=1).max()
        g["roll_mean_N"] = g["amount"].rolling(N, min_periods=5).mean()
        g["roll_std_N"]  = g["amount"].rolling(N, min_periods=5).std()
        g["amount_z"] = (g["amount"] - g["roll_mean_N"]) / g["roll_std_N"].replace(0, np.nan)
        g["amount_z"] = g["amount_z"].fillna(0)
        return g
    x = x.groupby(group_keys, group_keys=False).apply(add_roll) if group_keys else add_roll(x)

    x["hour"] = x["timestamp"].dt.hour
    x["dayofweek"] = x["timestamp"].dt.dayofweek
    x["is_weekend"] = (x["dayofweek"] >= 5).astype(int)

    alpha = 0.3
    def merch_feats(g):
        g = g.sort_values("timestamp").copy()
        top_share = []
        running_total = 0.0
        counts = {}
        for m in g["merchant"].values:
            running_total = running_total * (1 - alpha) + 1.0
            counts[m] = counts.get(m, 0.0) * (1 - alpha) + 1.0
            top_share.append(max(counts.values()) / max(running_total, 1e-6))
        g["merchant_conc_ewm"] = top_share
        g["repeat_small_charge"] = ((g["amount"] < 5) & (g["prev_txn_delta_min"] < 10)).astype(int)
        g["rapid_swipes"] = (g["prev_txn_delta_min"] < 2).astype(int)
        return g
    x = x.groupby(group_keys, group_keys=False).apply(merch_feats) if group_keys else merch_feats(x)

    def haversine_km(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1; dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2*np.arcsin(np.sqrt(a));  return 6371.0088 * c

    need_geo = all(c in x.columns for c in ["cust_lat","cust_long","merch_lat","merch_long"])
    if need_geo:
        geo_ok = x[["cust_lat","cust_long","merch_lat","merch_long"]].notna().all(axis=1)
        x["geo_missing"] = (~geo_ok).astype(int)
        x["geo_dist_km"] = 0.0
        mask = geo_ok.values
        if mask.any():
            x.loc[mask, "geo_dist_km"] = haversine_km(
                x.loc[mask,"cust_lat"].astype(float),
                x.loc[mask,"cust_long"].astype(float),
                x.loc[mask,"merch_lat"].astype(float),
                x.loc[mask,"merch_long"].astype(float),
            )
    else:
        x["geo_missing"] = 1
        x["geo_dist_km"] = 0.0
    return x

# ---------------- App setup ----------------
st.set_page_config(page_title="Smart Spending Tracker and Analyzer", layout="wide", page_icon="🔍")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Smart Spending Tracker and Analyzer")
st.markdown("**AI-Powered Categorization, Anomaly Detection, Risk Scoring & Predictive Insights**")

st.markdown(
    "- 📊 **Multi-dimensional anomaly detection** using Isolation Forest\n"
    "- 🎯 **Behavioral pattern analysis** and spending insights\n"
    "- 🔮 **Predictive analytics** for future spending trends\n"
    "- 🚨 **Advanced risk detection** with interpretable rules"
)

# ---------------- Load models ----------------
@st.cache_resource(show_spinner="Loading categorization model...")
def load_categorization_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner="Loading risk pipeline...")
def load_risk_pipeline():
    bundle = joblib.load(RISK_MODEL_PATH)
    return bundle["pipe"], bundle["feature_cols"]

# ---------------- Sidebar ----------------
st.sidebar.header("📁 Data Upload")
main_up = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"], key="main_up")
if main_up and "df_main" not in st.session_state:
    try:
        st.session_state["df_main"] = pd.read_csv(main_up)
        st.sidebar.success(f"✅ Loaded {len(st.session_state['df_main'])} transactions")
    except Exception as e:
        st.sidebar.error(f"❌ Could not read CSV: {e}")

# ---------------- Tabs ----------------
tab_insights, tab_anomaly, tab_risk, tab_budget, tab_patterns, tab_gloss = st.tabs([
    "📊 Insights & Categorization", 
    "🔍 Anomaly Detection",
    "🚨 Risk Detection", 
    "💰 Budgeting",
    "📈 Pattern Analysis",
    "📘 Glossary"
])

# ---------------- Insights Tab ----------------
with tab_insights:
    st.subheader("📊 Transaction Categorization & Spending Insights")

    try:
        pipe = load_categorization_model()
        st.success(f"✅ Categorization model loaded")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        pipe = None

    df_in = st.session_state.get("df_main")
    if df_in is None:
        up = st.file_uploader("Upload CSV for Insights", type=["csv"], key="insights_up")
        if up:
            try:
                df_in = pd.read_csv(up)
                st.session_state["df_main"] = df_in
            except Exception as e:
                st.error(f"❌ Error: {e}")
                df_in = None

    if df_in is None:
        st.info("👆 Upload a CSV to begin analysis")
    else:
        with st.expander("📋 Data Preview"):
            st.dataframe(df_in.head(20), use_container_width=True)

        try:
            df_aligned = align_to_canonical(df_in)
        except Exception as e:
            st.error(f"❌ Schema error: {e}")
            df_aligned = None

        if df_aligned is not None and pipe is not None:
            try:
                df_feat, dropped = build_features(df_aligned)
                if dropped > 0:
                    st.info(f"ℹ️ Cleaned {dropped} rows with missing data")
            except Exception as e:
                st.error(f"❌ Feature engineering error: {e}")
                df_feat = None

            if df_feat is not None:
                try:
                    preds = pipe.predict(df_feat)
                    st.session_state['predictions'] = preds
                    st.session_state['df_feat'] = df_feat
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
                    preds = None

                if preds is not None:
                    out = df_aligned.copy()
                    out["pred_category"] = preds
                    st.session_state['categorized_data'] = out

                    st.subheader("🎯 Prediction Results")
                    show_cols = [c for c in ["timestamp","merchant","amount","city","state","pred_category","category"] if c in out.columns]
                    st.dataframe(out[show_cols].head(50), use_container_width=True)

                    if "category" in out.columns:
                        try:
                            agree = (out["pred_category"] == out["category"]).mean()
                            st.metric("🎯 Model Accuracy", f"{agree:.2%}")
                        except Exception:
                            pass

                    facts = make_facts(out)
                    st.session_state['facts'] = facts

                    st.markdown("### 📈 Summary KPIs")
                    if not facts.empty:
                        total_spend = float(facts["amount"].sum())
                        tx_count = int(len(facts))
                        avg_ticket = float(facts["amount"].mean())
                        unique_merchants = int(facts["merchant"].nunique())
                        dmin, dmax = facts["timestamp"].min(), facts["timestamp"].max()
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Total Spend", f"₹{total_spend:,.0f}")
                        col2.metric("Transactions", f"{tx_count:,}")
                        col3.metric("Avg Ticket", f"₹{avg_ticket:,.0f}")
                        col4.metric("Unique Merchants", f"{unique_merchants}")
                        col5.metric("Days Tracked", f"{(dmax - dmin).days}")

                        # Interactive time series
                        st.markdown("### 📅 Spending Trends")
                        ts_freq = st.selectbox("Time grouping", ["Daily", "Weekly", "Monthly"], index=1)
                        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
                        
                        ts_data = facts.set_index("timestamp")["amount"].resample(freq_map[ts_freq]).agg(['sum', 'count', 'mean'])
                        
                        fig = make_subplots(rows=2, cols=1, 
                                          subplot_titles=('Total Spend', 'Transaction Count'),
                                          vertical_spacing=0.1)
                        
                        fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data['sum'], 
                                               mode='lines+markers', name='Spend',
                                               line=dict(color='#667eea', width=3)), row=1, col=1)
                        fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data['count'], 
                                               mode='lines+markers', name='Count',
                                               line=dict(color='#764ba2', width=3)), row=2, col=1)
                        
                        fig.update_layout(height=500, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                        # Category breakdown with plotly
                        st.markdown("### 🏷️ Category Analysis")
                        by_cat_amt = facts.groupby("pred_category")["amount"].sum().sort_values(ascending=False).head(10)
                        by_cat_cnt = facts["pred_category"].value_counts().head(10)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            fig_cat = px.pie(values=by_cat_amt.values, names=by_cat_amt.index, 
                                           title="Spend Distribution", hole=0.4)
                            st.plotly_chart(fig_cat, use_container_width=True)
                        with col_b:
                            fig_cnt = px.bar(x=by_cat_cnt.index, y=by_cat_cnt.values,
                                           title="Transaction Count", labels={'x':'Category', 'y':'Count'})
                            fig_cnt.update_traces(marker_color='#667eea')
                            st.plotly_chart(fig_cnt, use_container_width=True)

                        # Top merchants
                        st.markdown("### 🏪 Top Merchants")
                        cat_filter = st.selectbox("Filter by category", options=["(All)"] + sorted(facts["pred_category"].unique().tolist()))
                        fdf = facts if cat_filter == "(All)" else facts[facts["pred_category"] == cat_filter]
                        top_merchants = (
                            fdf.groupby("merchant")["amount"].agg(["sum","count"])
                            .rename(columns={"sum":"total_spend","count":"transactions"})
                            .sort_values("total_spend", ascending=False)
                            .head(15)
                        )
                        st.dataframe(top_merchants.style.format({"total_spend": "₹{:,.0f}"}), use_container_width=True)

                        # Recurring detection
                        st.markdown("### 🔄 Recurring Charges Detection")
                        rec = recurring_detection(facts)
                        if rec.empty:
                            st.info("No recurring patterns detected yet")
                        else:
                            rec_disp = rec.copy()
                            rec_disp["amount_cv"] = rec_disp["amount_cv"].round(3)
                            st.dataframe(rec_disp, use_container_width=True)
                            st.caption("💡 These merchants show consistent monthly billing patterns")

                        # High spend anomalies
                        st.markdown("### ⚠️ Unusually High Purchases")
                        anom = anomaly_high_spend(facts, top_n=10)
                        if anom.empty:
                            st.info("No statistical outliers detected")
                        else:
                            anom_disp = anom.copy()
                            anom_disp["z"] = anom_disp["z"].round(2)
                            st.dataframe(anom_disp, use_container_width=True)
                            st.caption("💡 Z-score shows how many standard deviations above category average")

                        # Download
                        st.markdown("### 💾 Export Data")
                        csv_buf = io.StringIO()
                        out[show_cols].to_csv(csv_buf, index=False)
                        st.download_button(
                            "📥 Download Predictions CSV",
                            csv_buf.getvalue(),
                            file_name="predictions.csv",
                            mime="text/csv",
                        )

# ---------------- NEW: Anomaly Detection Tab ----------------
with tab_anomaly:
    st.subheader("🔍 Multi-Dimensional Anomaly Detection")
    st.markdown("Advanced unsupervised learning to detect unusual transaction patterns")
    
    facts = st.session_state.get('facts')
    if facts is None or facts.empty:
        st.info("👆 Please run categorization in the Insights tab first")
    else:
        st.markdown("#### Detection Settings")
        col1, col2 = st.columns(2)
        with col1:
            contamination = st.slider("Expected anomaly rate (%)", 1, 15, 5) / 100
        with col2:
            st.metric("Total Transactions", len(facts))
        
        if st.button("🔍 Run Anomaly Detection", type="primary"):
            with st.spinner("Analyzing patterns..."):
                anomalies = detect_multi_dimensional_anomalies(facts, contamination=contamination)
                
            if anomalies.empty:
                st.warning("⚠️ No anomalies detected or insufficient data")
            else:
                st.success(f"✅ Detected {len(anomalies)} anomalous transactions")
                
                # Anomaly score distribution
                st.markdown("### 📊 Anomaly Score Distribution")
                fig = px.histogram(anomalies, x='anomaly_score', nbins=30,
                                 title="Distribution of Anomaly Scores",
                                 labels={'anomaly_score': 'Anomaly Score', 'count': 'Frequency'})
                fig.update_traces(marker_color='#e74c3c')
                st.plotly_chart(fig, use_container_width=True)
                
                # Reason breakdown
                st.markdown("### 🎯 Anomaly Reasons")
                reason_list = []
                for reasons in anomalies['anomaly_reason']:
                    reason_list.extend(reasons.split(','))
                reason_counts = pd.Series(reason_list).value_counts()
                
                fig_reasons = px.bar(x=reason_counts.index, y=reason_counts.values,
                                   title="Frequency of Anomaly Types",
                                   labels={'x': 'Reason', 'y': 'Count'})
                fig_reasons.update_traces(marker_color='#f39c12')
                st.plotly_chart(fig_reasons, use_container_width=True)
                
                # Top anomalies table
                st.markdown("### 🚨 Top Anomalies")
                display_anomalies = anomalies.head(20).copy()
                st.dataframe(display_anomalies.style.format({
                    'amount': '₹{:,.2f}',
                    'anomaly_score': '{:.3f}'
                }), use_container_width=True, height=400)
                
                # Time distribution of anomalies
                st.markdown("### 📅 Anomaly Timeline")
                anomalies_time = anomalies.copy()
                anomalies_time['date'] = pd.to_datetime(anomalies_time['timestamp']).dt.date
                daily_anomalies = anomalies_time.groupby('date').size().reset_index(name='count')
                
                fig_timeline = px.line(daily_anomalies, x='date', y='count',
                                     title="Daily Anomaly Count",
                                     markers=True)
                fig_timeline.update_traces(line_color='#e74c3c')
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Export anomalies
                csv_anom = anomalies.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Anomalies CSV",
                    csv_anom,
                    "anomalies.csv",
                    "text/csv"
                )

# ---------------- NEW: Pattern Analysis Tab ----------------
with tab_patterns:
    st.subheader("📈 Advanced Spending Pattern Analysis")
    st.markdown("Discover hidden behavioral patterns and get predictive insights")
    
    facts = st.session_state.get('facts')
    if facts is None or facts.empty:
        st.info("👆 Please run categorization in the Insights tab first")
    else:
        # Spending patterns
        st.markdown("### 🎯 Behavioral Patterns")
        patterns = analyze_spending_patterns(facts)
        
        if patterns:
            col1, col2, col3 = st.columns(3)
            with col1:
                ratio = patterns.get('weekend_vs_weekday_ratio', 1.0)
                st.metric("Weekend vs Weekday Ratio", f"{ratio:.2f}x",
                         help="Ratio of average weekend spend to weekday spend")
            with col2:
                diversity = patterns.get('merchant_diversity_score', 0)
                st.metric("Merchant Diversity Score", f"{diversity:.2f}",
                         help="Higher = more diverse merchants (entropy-based)")
            with col3:
                trend = patterns.get('spending_trend', 'stable')
                st.metric("Spending Trend", trend.capitalize(),
                         help="Overall spending trajectory")
            
            # Time-of-day heatmap
            st.markdown("### ⏰ Spending by Time of Day")
            if 'time_distribution' in patterns:
                time_dist = pd.DataFrame(patterns['time_distribution'])
                
                fig_time = px.bar(time_dist.T, barmode='group',
                                title="Spend Amount and Transaction Count by Time Period",
                                labels={'value': 'Amount / Count', 'index': 'Time Period'})
                st.plotly_chart(fig_time, use_container_width=True)
            
            # Day of week patterns
            st.markdown("### 📅 Day of Week Analysis")
            if 'dow_avg_amount' in patterns:
                dow_data = pd.DataFrame.from_dict(patterns['dow_avg_amount'], 
                                                  orient='index', 
                                                  columns=['avg_amount'])
                dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_data = dow_data.reindex([d for d in dow_order if d in dow_data.index])
                
                fig_dow = px.line(dow_data, markers=True,
                                title="Average Transaction Amount by Day",
                                labels={'index': 'Day', 'avg_amount': 'Average Amount'})
                fig_dow.update_traces(line_color='#667eea', marker_size=10)
                st.plotly_chart(fig_dow, use_container_width=True)
        
        # Predictive insights
        st.markdown("### 🔮 Predictive Insights")
        insights = generate_predictive_insights(facts)
        
        if insights:
            if 'projected_next_month' in insights:
                st.markdown("#### 💰 Next Month Projection")
                projected = insights['projected_next_month']
                current_month = facts[facts['month'] == facts['month'].max()]['amount'].sum()
                change = ((projected - current_month) / current_month * 100) if current_month > 0 else 0
                
                col1, col2 = st.columns(2)
                col1.metric("Projected Next Month", f"₹{projected:,.0f}", 
                           f"{change:+.1f}%")
                col2.metric("Current Month Total", f"₹{current_month:,.0f}")
            
            if 'growing_categories' in insights and insights['growing_categories']:
                st.markdown("#### 📈 Growing Spending Categories")
                st.info("These categories show increasing spend trends:")
                for cat in insights['growing_categories']:
                    st.write(f"• {cat}")
            
            if 'merchants_due_for_revisit' in insights and insights['merchants_due_for_revisit']:
                st.markdown("#### 🔄 Merchants You Might Visit Soon")
                st.caption("Based on your historical visit patterns:")
                revisit_df = pd.DataFrame(insights['merchants_due_for_revisit'])
                st.dataframe(revisit_df.style.format({
                    'avg_interval_days': '{:.0f}',
                    'days_since_last': '{:.0f}'
                }), use_container_width=True)
        
        # Category-merchant affinity matrix
        st.markdown("### 🔗 Category-Merchant Relationships")
        if 'pred_category' in facts.columns:
            top_cats = facts['pred_category'].value_counts().head(5).index
            top_merchants = facts['merchant'].value_counts().head(10).index
            
            affinity = facts[facts['pred_category'].isin(top_cats) & 
                           facts['merchant'].isin(top_merchants)].pivot_table(
                index='merchant',
                columns='pred_category',
                values='amount',
                aggfunc='sum',
                fill_value=0
            )
            
            fig_heatmap = px.imshow(affinity,
                                   title="Spend Heatmap: Top Merchants × Categories",
                                   labels=dict(x="Category", y="Merchant", color="Total Spend"),
                                   color_continuous_scale='Viridis')
            st.plotly_chart(fig_heatmap, use_container_width=True)

# ---------------- Risk Detection Tab ----------------
with tab_risk:
    st.subheader("🚨 Risk Detection (Unsupervised)")
    st.caption("Advanced anomaly scoring with interpretable rules")

    st.sidebar.subheader("⚙️ Risk Calibration")
    pct = st.sidebar.slider("Alert percentile", 95.0, 99.9, 99.0, 0.1)
    need_two_rules = st.sidebar.toggle("Require ≥2 rule hits", True)
    z_cut   = st.sidebar.slider("High z-score >", 2.0, 6.0, 4.0, 0.5)
    geo_km  = st.sidebar.slider("Geo jump km >", 200, 3000, 1000, 50)
    vel_mul = st.sidebar.slider("Velocity spike ×", 1.0, 10.0, 4.5, 0.5)

    use_ins_risk = st.toggle("Use Insights dataset", value=("df_main" in st.session_state))
    df_src_risk = st.session_state.get("df_main") if use_ins_risk else None
    
    if not use_ins_risk:
        risk_up = st.file_uploader("Upload CSV for Risk", type=["csv"], key="risk_up_tab")
        if risk_up:
            try:
                df_src_risk = pd.read_csv(risk_up)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if df_src_risk is None:
        st.info("👆 Provide dataset for risk analysis")
    else:
        df_risk = build_risk_features(df_src_risk)

        try:
            risk_pipe, feature_cols = load_risk_pipeline()
        except Exception as e:
            st.error(f"❌ Failed to load risk pipeline: {e}")
            risk_pipe, feature_cols = None, None

        if risk_pipe is not None:
            for c in feature_cols:
                if c not in df_risk.columns:
                    df_risk[c] = 0.0
            
            X = df_risk[feature_cols].fillna(0).values
            decision = risk_pipe.decision_function(X)
            risk_score = (-decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
            df_risk["risk_score"] = risk_score

            long_med = df_risk["roll_sum_N"].rolling(200, min_periods=50).median().fillna(0)
            rules = [
                ("high_z", df_risk["amount_z"] > z_cut),
                ("rapid_swipes", df_risk["rapid_swipes"] == 1),
                ("repeat_small", df_risk["repeat_small_charge"] == 1),
                ("geo_jump", df_risk["geo_dist_km"] > geo_km),
                ("velocity_spike", df_risk["roll_sum_N"] > (long_med * vel_mul)),
            ]
            reasons_arr = np.vstack([np.where(m, n, "") for n, m in rules]).T
            df_risk["reasons"] = [",".join([r for r in row if r]) if any(row) else "" for row in reasons_arr]

            thr = np.percentile(df_risk["risk_score"], pct)
            if need_two_rules:
                rule_count = (reasons_arr != "").sum(axis=1)
                rule_only = (rule_count >= 2)
                is_alert = (df_risk["risk_score"] >= thr) | rule_only
            else:
                is_alert = (df_risk["risk_score"] >= thr) | (df_risk["reasons"] != "")
            df_risk["is_alert"] = is_alert

            total = len(df_risk)
            alerts = int(is_alert.sum())
            alert_rate = 100.0 * alerts / max(total, 1)
            
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Total Transactions", f"{total:,}")
            k2.metric("🚨 Alerts", f"{alerts:,}")
            k3.metric("Alert Rate", f"{alert_rate:.2f}%")
            k4.metric("Risk Threshold", f"{thr:.3f}")

            # Risk score distribution
            st.markdown("### 📊 Risk Score Distribution")
            fig_risk_dist = px.histogram(df_risk, x='risk_score', nbins=50,
                                        title="Distribution of Risk Scores")
            fig_risk_dist.add_vline(x=thr, line_dash="dash", line_color="red",
                                   annotation_text="Threshold")
            st.plotly_chart(fig_risk_dist, use_container_width=True)

            st.markdown("### 🔍 Alert Explorer")
            with st.expander("🔧 Filter Alerts"):
                colA, colB, colC = st.columns(3)
                by_reason = colA.multiselect("Reason", 
                    ["high_z","rapid_swipes","repeat_small","geo_jump","velocity_spike"])
                min_amt = float(colB.number_input("Min amount", 0.0, 100000.0, 0.0, 10.0))
                city_like = colC.text_input("Search merchant/city", value="")
            
            fltr = df_risk["is_alert"].copy()
            if by_reason:
                fltr &= df_risk["reasons"].str.contains("|".join(by_reason), na=False)
            if city_like:
                city_like_l = city_like.strip().lower()
                has_city = "city" in df_risk.columns
                fltr &= (df_risk["merchant"].str.contains(city_like_l, na=False) |
                        (df_risk["city"].str.contains(city_like_l, na=False) if has_city else False))
            fltr &= (df_risk["amount"] >= min_amt)

            cols = [c for c in ["timestamp","merchant","category","amount","city","state","cc_num","risk_score","reasons"] 
                   if c in df_risk.columns]
            view = df_risk.loc[fltr, cols].sort_values("risk_score", ascending=False)
            st.dataframe(view.head(500), use_container_width=True, height=420)

            # Reason breakdown
            reason_counts = (df_risk.loc[df_risk["is_alert"], "reasons"]
                           .str.get_dummies(sep=",").sum().sort_values(ascending=False))
            st.markdown("### 🎯 Alert Reason Distribution")
            fig_reasons = px.bar(x=reason_counts.index, y=reason_counts.values,
                               title="Frequency of Risk Reasons",
                               labels={'x': 'Reason', 'y': 'Count'})
            fig_reasons.update_traces(marker_color='#e74c3c')
            st.plotly_chart(fig_reasons, use_container_width=True)

            # Downloads
            csv_alerts = view.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Alerts CSV", csv_alerts, 
                             "alerts.csv", "text/csv")

            manifest = {
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "total_rows": total,
                "alerts": alerts,
                "alert_rate_pct": round(alert_rate, 2),
                "threshold_percentile": float(pct),
                "numeric_threshold": float(thr),
                "rules": {
                    "need_two_rules": bool(need_two_rules),
                    "z_cut": float(z_cut),
                    "geo_km": int(geo_km),
                    "vel_mul": float(vel_mul)
                },
            }
            st.download_button("📥 Download Manifest JSON", 
                             json.dumps(manifest, indent=2).encode(),
                             "run_manifest.json", "application/json")

# ---------------- Budgeting Tab ----------------
with tab_budget:
    st.subheader("💰 Smart Budgeting")
    df_main = st.session_state.get("df_main")
    
    if df_main is None:
        st.info("👆 Upload CSV in sidebar to set budgets")
    else:
        try:
            df_aligned = align_to_canonical(df_main)
            df_feat, _ = build_features(df_aligned)
            pipe = load_categorization_model()
            preds = pipe.predict(df_feat)
            out = df_aligned.copy()
            out["pred_category"] = preds
        except Exception:
            out = df_aligned.copy()
            if "category" in out.columns:
                out["pred_category"] = out["category"].astype(str)
            else:
                out["pred_category"] = "uncategorized"
        
        facts = make_facts(out)

        if facts.empty:
            st.info("No data available after cleaning")
        else:
            months = sorted(facts["month"].unique().tolist())
            sel_month = st.selectbox("📅 Select month", options=months, 
                                    index=len(months)-1 if months else 0)
            mdf = facts[facts["month"] == sel_month]
            
            if mdf.empty:
                st.info("No data for selected month")
            else:
                all_cats = sorted(mdf["pred_category"].unique().tolist())
                default_cats = (
                    mdf.groupby("pred_category")["amount"].sum()
                    .sort_values(ascending=False).head(5).index.tolist()
                )
                chosen = st.multiselect("Choose up to 5 categories", 
                                       options=all_cats, 
                                       default=default_cats, 
                                       max_selections=5)
                
                if not chosen:
                    st.caption("Select categories to define budgets")
                else:
                    cols = st.columns(len(chosen))
                    for i, cat in enumerate(chosen):
                        key = f"budget_{cat}"
                        if key not in st.session_state:
                            st.session_state[key] = 0.0
                        cols[i].number_input(f"Budget for {cat}", 
                                           min_value=0.0, 
                                           value=float(st.session_state[key]),
                                           step=100.0, key=key)

                    def clear_budgets():
                        for cat in chosen:
                            st.session_state[f"budget_{cat}"] = 0.0
                    st.button("🗑️ Clear all budgets", on_click=clear_budgets)

                    by_cat_curr = mdf.groupby("pred_category")["amount"].sum()
                    rows = []
                    for cat in chosen:
                        budget = float(st.session_state[f"budget_{cat}"])
                        actual = float(by_cat_curr.get(cat, 0.0))
                        variance = budget - actual
                        pct_used = (actual / budget * 100) if budget > 0 else 0
                        rows.append({
                            "category": cat,
                            "budget": budget,
                            "actual": actual,
                            "variance": variance,
                            "pct_used": pct_used
                        })
                    
                    if rows:
                        bdf = pd.DataFrame(rows)
                        
                        # Visual budget progress
                        st.markdown("### 📊 Budget Progress")
                        for _, row in bdf.iterrows():
                            st.write(f"**{row['category']}**")
                            progress = min(row['pct_used'] / 100, 1.0)
                            color = "green" if progress < 0.8 else ("orange" if progress < 1.0 else "red")
                            st.progress(progress)
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Budget", f"₹{row['budget']:,.0f}")
                            col2.metric("Spent", f"₹{row['actual']:,.0f}")
                            col3.metric("Remaining", f"₹{row['variance']:,.0f}",
                                      delta=f"{row['pct_used']:.0f}% used")
                            st.write("---")
                        
                        # Summary table
                        bdf.loc[len(bdf)] = {
                            "category": "TOTAL",
                            "budget": bdf["budget"].sum(),
                            "actual": bdf["actual"].sum(),
                            "variance": bdf["variance"].sum(),
                            "pct_used": (bdf["actual"].sum() / bdf["budget"].sum() * 100) if bdf["budget"].sum() > 0 else 0
                        }
                        
                        st.markdown("### 📋 Budget Summary Table")
                        styled = (
                            bdf.style
                            .background_gradient(subset=["variance"], cmap="RdYlGn")
                            .format({
                                "budget": "₹{:,.0f}",
                                "actual": "₹{:,.0f}",
                                "variance": "₹{:,.0f}",
                                "pct_used": "{:.1f}%"
                            })
                        )
                        st.table(styled)
                        st.caption("💡 Variance = budget − actual (positive means under budget)")

# ---------------- Glossary Tab ----------------
with tab_gloss:
    st.header("📘 Glossary & Concepts")
    st.caption("Plain-English explanations of key concepts")
    
    with st.expander("🔍 **Anomaly Detection**"):
        st.markdown("""
        **What it is:** Identifies transactions that deviate significantly from normal patterns.
        
        **How it works:** Uses machine learning (Isolation Forest) to analyze multiple dimensions 
        like amount, time, merchant frequency, and day-of-week patterns simultaneously.
        
        **Example:** A ₹5,000 purchase at 3 AM from a merchant you've never used before 
        would likely be flagged as anomalous.
        """)
    
    
    with st.expander("📊 **Isolation Forest**"):
        st.markdown("""
        **What it is:** An unsupervised ML algorithm specifically designed for anomaly detection.
        
        **How it works:** Randomly creates decision trees that isolate data points. 
        Anomalies are easier to isolate (require fewer splits), so they have shorter paths.
        
        **Why it's good:** Fast, works well with high-dimensional data, and doesn't require 
        labeled anomalies for training.
        """)
    
    with st.expander("⚡ **Risk Scoring**"):
        st.markdown("""
        **Anomaly (unsupervised):** Identifies unusual patterns without needing fraud labels.
        
        **IsolationForest score:** Continuous score where lower = more anomalous. 
        We convert this to risk_score where higher = riskier.
        
        **Percentile threshold:** Flags top X% as alerts. Setting 99th percentile 
        means ~1% of transactions become alerts.
        
        **Rule-based promotion:** Even if score is below threshold, multiple rule hits 
        (e.g., high z-score + geo jump) can still trigger an alert.
        """)
    
    with st.expander("📈 **Z-Score**"):
        st.markdown("""
        **What it is:** Measures how many standard deviations a value is from the mean.
        
        **Formula:** z = (value - mean) / standard_deviation
        
        **Example:** If average transaction is ₹500 with std dev ₹200, 
        a ₹1,300 transaction has z = (1300-500)/200 = 4.0
        
        **Interpretation:** z > 3 is typically considered an outlier.
        """)
    
    with st.expander("🔄 **Recurring Charges**"):
        st.markdown("""
        **Detection criteria:**
        - Minimum 3 transactions to the same merchant
        - Median interval between 26-33 days (roughly monthly)
        - Low amount variation (coefficient of variation < 0.2)
        
        **Example:** Netflix subscription at ₹499 every 30 days would be detected.
        """)
    
    with st.expander("🎯 **Merchant Diversity Score**"):
        st.markdown("""
        **What it is:** Entropy-based measure of how spread out spending is across merchants.
        
        **Scale:** 
        - 0 = All spending at one merchant (no diversity)
        - Higher = More diverse spending patterns
        - Typical range: 2-4 for normal users
        
        **Why it matters:** Sudden drops might indicate account compromise; 
        steady high values suggest healthy financial habits.
        """)
    
    with st.expander("🌍 **Geo Jump**"):
        st.markdown("""
        **What it is:** Flags transactions where customer→merchant distance is 
        unusually large given the time between transactions.
        
        **Example:** Transaction in Mumbai at 2 PM, then Delhi at 2:30 PM 
        (physically impossible without flight).
        
        **Use case:** Indicates potential card cloning or account takeover.
        """)
    
    with st.expander("⚡ **Velocity Spike**"):
        st.markdown("""
        **What it is:** Detects when recent spending rate suddenly exceeds historical baseline.
        
        **How it works:** Compares rolling 20-transaction sum to 200-transaction median. 
        Alert if current rate > baseline × multiplier (e.g., 4.5x).
        
        **Example:** User typically spends ₹10k per 20 transactions, suddenly spends 
        ₹50k in last 20 transactions → velocity spike.
        """)
    
    with st.expander("🎨 **Feature Engineering**"):
        st.markdown("""
        **What it is:** Creating new variables from raw data to help ML models learn better.
        
        **Examples in this app:**
        - `log_amount`: log(1 + amount) to handle skewed distributions
        - `day_of_week`: Extract from timestamp (0=Monday, 6=Sunday)
        - `hour`: Time of day when transaction occurred
        - `amount_bin`: Discretized amount ranges for pattern detection
        """)
    
    with st.expander("🔮 **Predictive Insights**"):
        st.markdown("""
        **Next month projection:** Simple moving average of recent 3 months.
        
        **Growing categories:** Categories where latest month > 3 months ago.
        
        **Merchants due for revisit:** Based on historical visit intervals. 
        If you visit Starbucks every 7 days on average, and it's been 6 days, 
        it predicts you might go soon.
        
        **Limitations:** These are statistical projections, not guarantees. 
        Real spending depends on many factors.
        """)
    
    st.markdown("---")
    st.markdown("### 💡 Pro Tips")
    st.info("""
    **For best results:**
    1. Upload at least 3 months of transaction history
    2. Ensure consistent date formats (YYYY-MM-DD recommended)
    3. Include merchant names for better categorization
    4. For risk detection, include geo coordinates if available
    5. Regularly review anomaly alerts to tune sensitivity
    """)
    
    st.markdown("### 🛡️ Data Privacy")
    st.success("""
    **Your data is safe:**
    - All processing happens locally in your browser session
    - No data is sent to external servers
    - Data is cleared when you close the tab
    - Models are pre-trained and don't learn from your data
    """)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("💡 **Tip:** For reproducible results across sessions, save and reuse the same bin edges for amount discretization. "
          "Export your results regularly for record-keeping.")


import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# So `streamlit run app.py` works from a plain checkout, without pip install -e .
SRC = Path(__file__).resolve().parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spending_analyzer import (
    ModelLoadError,
    RiskThresholds,
    SchemaError,
    align_to_canonical,
    analyze_spending_patterns,
    anomaly_high_spend,
    apply_rules,
    build_features,
    build_risk_features,
    detect_multi_dimensional_anomalies,
    generate_predictive_insights,
    load_amount_binning,
    load_categorization_pipeline,
    load_risk_bundle,
    make_facts,
    recurring_detection,
    score_risk,
)

RS = "₹"
PURPLE = "#667eea"
DEEP_PURPLE = "#764ba2"
RED = "#e74c3c"
ORANGE = "#f39c12"

st.set_page_config(page_title="Smart Spending Tracker and Analyzer", layout="wide", page_icon="🔍")


# ---------------- Cached work ----------------
@st.cache_resource(show_spinner="Loading categorization model...")
def get_pipe():
    return load_categorization_pipeline()


@st.cache_resource(show_spinner="Loading risk pipeline...")
def get_risk_bundle():
    return load_risk_bundle()


@st.cache_data(show_spinner="Reading file...")
def read_upload(payload):
    return pd.read_csv(io.BytesIO(payload))


@st.cache_data(show_spinner="Categorizing transactions...")
def categorize(raw):
    feats, dropped, _ = build_features(align_to_canonical(raw))
    if feats.empty:
        return feats, dropped
    out = feats.drop(columns=["log_amount", "amount_bin"], errors="ignore").copy()
    out["pred_category"] = get_pipe().predict(feats)
    return out, dropped


@st.cache_data(show_spinner="Scoring risk...")
def score_transactions(raw):
    feats = build_risk_features(raw)
    feats["risk_score"] = score_risk(feats, get_risk_bundle())
    return feats


def rs(v):
    return f"{RS}{v:,.0f}"


# ---------------- Sidebar ----------------
def load_dataset():
    st.sidebar.header("📁 Data Upload")
    up = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"], key="main_up")
    if up is None:
        return st.session_state.get("df_main")

    # Re-read whenever a different file shows up, otherwise only the first
    # upload of a session ever gets used.
    fingerprint = (up.name, up.size)
    if st.session_state.get("upload_fingerprint") != fingerprint:
        try:
            st.session_state["df_main"] = read_upload(up.getvalue())
            st.session_state["upload_fingerprint"] = fingerprint
        except Exception as e:
            st.sidebar.error(f"❌ Could not read CSV: {e}")
            return st.session_state.get("df_main")

    df = st.session_state.get("df_main")
    if df is not None:
        st.sidebar.success(f"✅ Loaded {len(df):,} rows from {up.name}")
    return df


st.title("🔍 Smart Spending Tracker and Analyzer")
st.markdown("**AI-powered categorization, anomaly detection, risk scoring and forecasting**")

df_main = load_dataset()
binning = load_amount_binning()
st.sidebar.caption(f"Amount bins: {binning.n_bins} · source: {Path(binning.source).name}")

tab_insights, tab_anomaly, tab_risk, tab_budget, tab_patterns, tab_gloss = st.tabs([
    "📊 Insights & Categorization",
    "🔍 Anomaly Detection",
    "🚨 Risk Detection",
    "💰 Budgeting",
    "📈 Pattern Analysis",
    "📘 Glossary",
])


# ---------------- Insights ----------------
with tab_insights:
    st.subheader("📊 Transaction Categorization & Spending Insights")

    if df_main is None:
        st.info("👆 Upload a transactions CSV in the sidebar to begin.")
    else:
        with st.expander("📋 Data preview"):
            st.dataframe(df_main.head(20), use_container_width=True)

        out, dropped = None, 0
        try:
            out, dropped = categorize(df_main)
        except (SchemaError, ModelLoadError) as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error during categorization: {e}")

        if out is not None and out.empty:
            st.warning("Nothing left after cleaning. Check the timestamp and amount columns.")
        elif out is not None:
            if dropped:
                st.info(f"ℹ️ Dropped {dropped:,} row(s) with unusable timestamps or amounts.")

            facts = make_facts(out)
            st.session_state["facts"] = facts

            show_cols = [c for c in ["timestamp", "merchant", "amount", "city", "state",
                                     "pred_category", "category"] if c in out.columns]
            st.markdown("### 🎯 Prediction results")
            st.dataframe(out[show_cols].head(50), use_container_width=True)

            if "category" in out.columns:
                agree = (out["pred_category"] == out["category"]).mean()
                st.metric("🎯 Agreement with supplied labels", f"{agree:.2%}",
                          help="Only meaningful when the file already has ground-truth categories.")

            st.markdown("### 📈 Summary")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total spend", rs(facts["amount"].sum()))
            c2.metric("Transactions", f"{len(facts):,}")
            c3.metric("Avg ticket", rs(facts["amount"].mean()))
            c4.metric("Unique merchants", f"{facts['merchant'].nunique():,}")
            c5.metric("Days tracked", f"{(facts['timestamp'].max() - facts['timestamp'].min()).days:,}")

            st.markdown("### 📅 Spending trends")
            grouping = st.selectbox("Time grouping", ["Daily", "Weekly", "Monthly"], index=1)
            ts = (facts.set_index("timestamp")["amount"]
                  .resample({"Daily": "D", "Weekly": "W", "Monthly": "ME"}[grouping])
                  .agg(["sum", "count"]))

            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=("Total spend", "Transaction count"),
                                vertical_spacing=0.12)
            fig.add_trace(go.Scatter(x=ts.index, y=ts["sum"], mode="lines+markers",
                                     name="Spend", line={"color": PURPLE, "width": 3}), row=1, col=1)
            fig.add_trace(go.Scatter(x=ts.index, y=ts["count"], mode="lines+markers",
                                     name="Count", line={"color": DEEP_PURPLE, "width": 3}), row=2, col=1)
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🏷️ Category analysis")
            by_amt = facts.groupby("pred_category")["amount"].sum().sort_values(ascending=False).head(10)
            by_cnt = facts["pred_category"].value_counts().head(10)
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(px.pie(values=by_amt.to_numpy(), names=by_amt.index,
                                       title="Spend distribution", hole=0.4),
                                use_container_width=True)
            with col_b:
                bar = px.bar(x=by_cnt.index, y=by_cnt.to_numpy(), title="Transaction count",
                             labels={"x": "Category", "y": "Count"})
                bar.update_traces(marker_color=PURPLE)
                st.plotly_chart(bar, use_container_width=True)

            st.markdown("### 🏪 Top merchants")
            cat_filter = st.selectbox("Filter by category",
                                      ["(All)"] + sorted(facts["pred_category"].unique().tolist()))
            fdf = facts if cat_filter == "(All)" else facts[facts["pred_category"] == cat_filter]
            top_merchants = (fdf.groupby("merchant")["amount"]
                             .agg(total_spend="sum", transactions="count")
                             .sort_values("total_spend", ascending=False).head(15))
            st.dataframe(top_merchants.style.format({"total_spend": RS + "{:,.0f}"}),
                         use_container_width=True)

            st.markdown("### 🔄 Recurring charges")
            rec = recurring_detection(facts)
            if rec.empty:
                st.info("No monthly subscriptions found. Needs 3+ charges to the same merchant, "
                        "26-33 days apart, at a steady amount.")
            else:
                st.dataframe(rec.style.format({"amount_cv": "{:.3f}",
                                               "median_interval_days": "{:.1f}"}),
                             use_container_width=True)
                st.caption("💡 Next dates are extrapolated from the median interval.")

            st.markdown("### ⚠️ Unusually high purchases")
            anom = anomaly_high_spend(facts, top_n=10)
            if anom.empty:
                st.info("No statistical outliers detected.")
            else:
                st.dataframe(anom.style.format({"z": "{:.2f}"}), use_container_width=True)
                st.caption("💡 z = standard deviations above that category's average.")

            st.markdown("### 💾 Export")
            st.download_button("📥 Download predictions CSV",
                               out[show_cols].to_csv(index=False).encode("utf-8"),
                               file_name="predictions.csv", mime="text/csv")


# ---------------- Anomaly detection ----------------
with tab_anomaly:
    st.subheader("🔍 Multi-dimensional anomaly detection")
    st.caption("Isolation Forest over time, amount, merchant frequency and category features.")

    facts = st.session_state.get("facts")
    if facts is None or facts.empty:
        st.info("👆 Run categorization on the Insights tab first.")
    else:
        c1, c2 = st.columns(2)
        contamination = c1.slider("Expected anomaly rate (%)", 1, 15, 5) / 100
        c2.metric("Transactions available", f"{len(facts):,}")

        if st.button("🔍 Run anomaly detection", type="primary"):
            with st.spinner("Analyzing patterns..."):
                st.session_state["anomaly_result"] = detect_multi_dimensional_anomalies(
                    facts, contamination=contamination)

        res = st.session_state.get("anomaly_result")
        if res is None:
            st.caption("Press the button to score the current dataset.")
        elif res.total_anomalies == 0:
            st.warning("⚠️ No anomalies found, or fewer than 10 usable transactions.")
        else:
            pct = 100 * res.total_anomalies / max(res.total_rows, 1)
            st.success(f"✅ {res.total_anomalies:,} anomalous transactions "
                       f"({pct:.1f}% of {res.total_rows:,})")
            if res.is_truncated:
                st.caption(f"Showing the {len(res.anomalies)} highest-scoring below.")

            st.markdown("### 📊 Score distribution")
            hist = px.histogram(res.scores.to_frame("anomaly_score"), x="anomaly_score", nbins=40,
                                title="Anomaly scores across all transactions",
                                labels={"anomaly_score": "Anomaly score"})
            hist.update_traces(marker_color=RED)
            st.plotly_chart(hist, use_container_width=True)

            st.markdown("### 🎯 Why these were flagged")
            reasons = (res.anomalies["anomaly_reason"].str.get_dummies(sep=",")
                       .sum().sort_values(ascending=False))
            rfig = px.bar(x=reasons.index, y=reasons.to_numpy(),
                          title="Frequency of anomaly reasons", labels={"x": "Reason", "y": "Count"})
            rfig.update_traces(marker_color=ORANGE)
            st.plotly_chart(rfig, use_container_width=True)

            st.markdown("### 🚨 Top anomalies")
            st.dataframe(res.anomalies.style.format({"amount": RS + "{:,.2f}",
                                                     "anomaly_score": "{:.3f}"}),
                         use_container_width=True, height=400)

            st.markdown("### 📅 Anomaly timeline")
            daily = (res.anomalies.assign(date=res.anomalies["timestamp"].dt.date)
                     .groupby("date").size().reset_index(name="count"))
            line = px.line(daily, x="date", y="count", title="Anomalies per day", markers=True)
            line.update_traces(line_color=RED)
            st.plotly_chart(line, use_container_width=True)

            st.download_button("📥 Download anomalies CSV",
                               res.anomalies.to_csv(index=False).encode("utf-8"),
                               "anomalies.csv", "text/csv")


# ---------------- Risk detection ----------------
with tab_risk:
    st.subheader("🚨 Risk Detection (Unsupervised)")
    st.caption("Anomaly scoring plus interpretable rules you can tune.")

    st.sidebar.subheader("⚙️ Risk Calibration")
    thresholds = RiskThresholds(
        percentile=st.sidebar.slider("Alert percentile", 95.0, 99.9, 99.0, 0.1),
        require_two_rules=st.sidebar.toggle("Require ≥2 rule hits", True),
        z_cut=st.sidebar.slider("High z-score >", 2.0, 6.0, 4.0, 0.5),
        geo_km=st.sidebar.slider("Geo jump km >", 200, 3000, 1000, 50),
        velocity_multiplier=st.sidebar.slider("Velocity spike ×", 1.0, 10.0, 4.5, 0.5),
    )

    if df_main is None:
        st.info("👆 Upload a transactions CSV in the sidebar.")
    else:
        scored = None
        try:
            scored = apply_rules(score_transactions(df_main), thresholds)
        except (ValueError, ModelLoadError) as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error during risk scoring: {e}")

        if scored is not None:
            thr = scored.attrs["risk_threshold"]
            alerts = int(scored["is_alert"].sum())
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Transactions", f"{len(scored):,}")
            k2.metric("🚨 Alerts", f"{alerts:,}")
            k3.metric("Alert Rate", f"{100 * alerts / max(len(scored), 1):.2f}%")
            k4.metric("Risk Threshold", f"{thr:.3f}")

            st.markdown("### 📊 Risk score distribution")
            dist = px.histogram(scored, x="risk_score", nbins=50, title="Risk scores")
            dist.add_vline(x=thr, line_dash="dash", line_color="red", annotation_text="Threshold")
            st.plotly_chart(dist, use_container_width=True)

            st.markdown("### 🔍 Alert explorer")
            with st.expander("🔧 Filter alerts"):
                f1, f2, f3 = st.columns(3)
                by_reason = f1.multiselect("Reason", ["high_z", "rapid_swipes", "repeat_small",
                                                      "geo_jump", "velocity_spike"])
                min_amt = float(f2.number_input("Min amount", 0.0, value=0.0, step=10.0))
                search = f3.text_input("Search merchant/city", value="")

            keep = scored["is_alert"] & (scored["amount"] >= min_amt)
            if by_reason:
                keep &= scored["reasons"].str.contains("|".join(by_reason), na=False)
            if search.strip():
                needle = search.strip().lower()
                hit = scored["merchant"].str.contains(needle, na=False, regex=False)
                if "city" in scored.columns:
                    hit |= scored["city"].str.contains(needle, na=False, regex=False)
                keep &= hit

            cols = [c for c in ["timestamp", "merchant", "category", "amount", "city", "state",
                                "cc_num", "risk_score", "reasons"] if c in scored.columns]
            view = scored.loc[keep, cols].sort_values("risk_score", ascending=False)
            st.caption(f"{len(view):,} matching alert(s), showing up to 500.")
            st.dataframe(view.head(500), use_container_width=True, height=420)

            st.markdown("### 🎯 Alert reason distribution")
            reason_counts = (scored.loc[scored["is_alert"], "reasons"].str.get_dummies(sep=",")
                             .sum().sort_values(ascending=False))
            if reason_counts.empty:
                st.info("These alerts came from the model score alone, no rule fired.")
            else:
                rbar = px.bar(x=reason_counts.index, y=reason_counts.to_numpy(),
                              title="Frequency of risk reasons", labels={"x": "Reason", "y": "Count"})
                rbar.update_traces(marker_color=RED)
                st.plotly_chart(rbar, use_container_width=True)

            d1, d2 = st.columns(2)
            d1.download_button("📥 Download alerts CSV", view.to_csv(index=False).encode("utf-8"),
                               "alerts.csv", "text/csv")
            manifest = {
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0)
                                .isoformat().replace("+00:00", "Z"),
                "total_rows": int(len(scored)),
                "alerts": alerts,
                "alert_rate_pct": round(100 * alerts / max(len(scored), 1), 2),
                "threshold_percentile": thresholds.percentile,
                "numeric_threshold": float(thr),
                "rules": {
                    "require_two_rules": thresholds.require_two_rules,
                    "z_cut": thresholds.z_cut,
                    "geo_km": thresholds.geo_km,
                    "velocity_multiplier": thresholds.velocity_multiplier,
                },
            }
            d2.download_button("📥 Download Manifest JSON", json.dumps(manifest, indent=2).encode(),
                               "run_manifest.json", "application/json")


# ---------------- Budgeting ----------------
with tab_budget:
    st.subheader("💰 Smart Budgeting")

    facts = st.session_state.get("facts")
    if facts is None or facts.empty:
        st.info("👆 Run categorization on the Insights tab first.")
    else:
        months = sorted(facts["month"].unique().tolist())
        sel_month = st.selectbox("📅 Select month", options=months, index=len(months) - 1)
        mdf = facts[facts["month"] == sel_month]

        by_cat = mdf.groupby("pred_category")["amount"].sum()
        default_cats = by_cat.sort_values(ascending=False).head(5).index.tolist()
        chosen = st.multiselect("Choose up to 5 categories",
                                options=sorted(mdf["pred_category"].unique().tolist()),
                                default=default_cats, max_selections=5)

        if not chosen:
            st.caption("Select categories to define budgets.")
        else:
            cols = st.columns(len(chosen))
            for col, cat in zip(cols, chosen, strict=True):
                col.number_input(f"Budget for {cat}", min_value=0.0, step=100.0, key=f"budget_{cat}")

            def clear_budgets():
                for cat in chosen:
                    st.session_state[f"budget_{cat}"] = 0.0

            st.button("🗑️ Clear all budgets", on_click=clear_budgets)

            rows = []
            for cat in chosen:
                budget = float(st.session_state.get(f"budget_{cat}", 0.0))
                actual = float(by_cat.get(cat, 0.0))
                rows.append({
                    "category": cat,
                    "budget": budget,
                    "actual": actual,
                    "variance": budget - actual,
                    "pct_used": (actual / budget * 100) if budget > 0 else 0.0,
                })
            bdf = pd.DataFrame(rows)

            st.markdown("### 📊 Budget progress")
            for row in rows:
                st.write(f"**{row['category']}**")
                st.progress(min(row["pct_used"] / 100, 1.0))
                b1, b2, b3 = st.columns(3)
                b1.metric("Budget", rs(row["budget"]))
                b2.metric("Spent", rs(row["actual"]))
                b3.metric("Remaining", rs(row["variance"]),
                          delta=f"{row['pct_used']:.0f}% used", delta_color="inverse")
                st.divider()

            total = {
                "category": "TOTAL",
                "budget": bdf["budget"].sum(),
                "actual": bdf["actual"].sum(),
                "variance": bdf["variance"].sum(),
                "pct_used": (bdf["actual"].sum() / bdf["budget"].sum() * 100
                             if bdf["budget"].sum() > 0 else 0.0),
            }
            summary = pd.concat([bdf, pd.DataFrame([total])], ignore_index=True)

            st.markdown("### 📋 Budget summary table")
            st.dataframe(summary.style.format({"budget": RS + "{:,.0f}",
                                               "actual": RS + "{:,.0f}",
                                               "variance": RS + "{:,.0f}",
                                               "pct_used": "{:.1f}%"}),
                         use_container_width=True)
            st.caption("💡 Variance = budget − actual, so positive means under budget.")


# ---------------- Pattern analysis ----------------
with tab_patterns:
    st.subheader("📈 Advanced Spending Pattern Analysis")

    facts = st.session_state.get("facts")
    if facts is None or facts.empty:
        st.info("👆 Run categorization on the Insights tab first.")
    else:
        patterns = analyze_spending_patterns(facts)

        st.markdown("### 🎯 Behavioural patterns")
        p1, p2, p3 = st.columns(3)
        ratio = patterns.get("weekend_vs_weekday_ratio")
        p1.metric("Weekend vs weekday", f"{ratio:.2f}×" if ratio else "n/a",
                  help="Average weekend spend over average weekday spend.")
        p2.metric("Merchant diversity", f"{patterns.get('merchant_diversity_score', 0):.2f}",
                  help="Entropy of merchant shares, higher means more spread out.")
        p3.metric("Spending trend", patterns.get("spending_trend", "n/a").capitalize(),
                  help="Last three weeks against the first three, needs 6+ weeks of data.")

        if patterns.get("time_distribution"):
            st.markdown("### ⏰ Spending by time of day")
            st.plotly_chart(px.bar(pd.DataFrame(patterns["time_distribution"]), barmode="group",
                                   title="Spend and transaction count by period",
                                   labels={"value": "Amount / count", "index": "Time period"}),
                            use_container_width=True)

        if patterns.get("dow_avg_amount"):
            st.markdown("### 📅 Day of week analysis")
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow = (pd.DataFrame.from_dict(patterns["dow_avg_amount"], orient="index",
                                          columns=["avg_amount"])
                   .reindex([d for d in order if d in patterns["dow_avg_amount"]]))
            dfig = px.line(dow, markers=True, title="Average transaction amount by day",
                           labels={"index": "Day", "value": "Average amount"})
            dfig.update_traces(line_color=PURPLE, marker_size=10)
            st.plotly_chart(dfig, use_container_width=True)

        st.markdown("### 🔮 Predictive insights")
        insights = generate_predictive_insights(facts)
        if not insights:
            st.info("Needs at least 30 transactions before projecting anything.")
        else:
            st.caption(f"Projected from data up to {insights['as_of']:%d %b %Y}.")
            if "projected_next_month" in insights:
                latest = facts["month"].max()
                current = float(facts.loc[facts["month"] == latest, "amount"].sum())
                projected = insights["projected_next_month"]
                change = ((projected - current) / current * 100) if current > 0 else 0.0
                m1, m2 = st.columns(2)
                m1.metric("Projected next month", rs(projected), f"{change:+.1f}%")
                m2.metric(f"Latest month ({latest})", rs(current))

            if insights.get("growing_categories"):
                st.markdown("#### 📈 Categories trending up")
                for cat in insights["growing_categories"]:
                    st.write(f"• {cat}")

            if insights.get("merchants_due_for_revisit"):
                st.markdown("#### 🔄 Merchants you might visit soon")
                st.dataframe(
                    pd.DataFrame(insights["merchants_due_for_revisit"]).style.format(
                        {"avg_interval_days": "{:.1f}", "days_since_last": "{:.1f}"}),
                    use_container_width=True)

        st.markdown("### 🔗 Category-merchant relationships")
        top_cats = facts["pred_category"].value_counts().head(5).index
        top_merchants = facts["merchant"].value_counts().head(10).index
        affinity = (facts[facts["pred_category"].isin(top_cats)
                          & facts["merchant"].isin(top_merchants)]
                    .pivot_table(index="merchant", columns="pred_category", values="amount",
                                 aggfunc="sum", fill_value=0))
        if affinity.empty:
            st.info("Not enough overlap between the top merchants and the top categories.")
        else:
            st.plotly_chart(px.imshow(affinity, title="Spend heatmap: top merchants × categories",
                                      labels={"x": "Category", "y": "Merchant",
                                              "color": "Total spend"},
                                      color_continuous_scale="Viridis"),
                            use_container_width=True)


# ---------------- Glossary ----------------
with tab_gloss:
    st.header("📘 Glossary & Concepts")
    st.caption("Plain-English explanations of what the tabs above are doing.")

    GLOSSARY = {
        "🔍 **Anomaly Detection**": """
        **What it is:** flags transactions that don't fit your usual pattern.

        **How it works:** an Isolation Forest looks at amount, hour, day of week,
        how often you use that merchant and how busy that day was, all together.

        **Example:** a ₹5,000 purchase at 3 AM at a merchant you've used twice.
        """,
        "📊 **Isolation Forest**": """
        **What it is:** an unsupervised algorithm built for outlier detection.

        **How it works:** it builds random trees that keep splitting the data.
        Outliers get isolated in fewer splits, so their average path is shorter.

        **Why it suits this:** no fraud labels needed, and it handles a lot of
        features at once.
        """,
        "⚡ **Risk Scoring**": """
        **Model score:** the detector's decision function, rescaled to 0-1 with
        higher meaning riskier.

        **Percentile threshold:** the top X% of scores become alerts. At the 99th
        percentile roughly 1% of transactions get flagged.

        **Rule promotion:** something below the threshold can still alert if
        enough rules fire, say a high z-score together with a geo jump.
        """,
        "📈 **Z-Score**": """
        **Formula:** z = (value − mean) / standard deviation.

        **Example:** with a ₹500 average and ₹200 standard deviation, a ₹1,300
        charge has z = 4.0.

        **Rule of thumb:** anything past |z| > 3 is worth a look.
        """,
        "🔄 **Recurring Charges**": """
        **Criteria:** at least 3 charges to the same merchant, a median gap of
        26-33 days, and a coefficient of variation under 0.2 on the amount.

        **Example:** a ₹499 subscription billed every 30 days.
        """,
        "🎯 **Merchant Diversity Score**": """
        **What it is:** entropy of how your spending spreads across merchants.

        **Scale:** 0 means everything at one merchant. Normal everyday spending
        usually lands somewhere around 2-4.
        """,
        "🌍 **Geo Jump**": """
        **What it is:** great-circle distance between the cardholder location and
        the merchant location, flagged past the threshold you set.

        **Needs:** `lat`, `long`, `merch_lat` and `merch_long` in the file.
        """,
        "⚡ **Velocity Spike**": """
        **What it is:** rolling 20-transaction spend compared against the rolling
        200-transaction median.

        **Note:** it stays quiet until there are 50 transactions of history, so
        the start of a file isn't flagged just for having no baseline yet.
        """,
        "🎨 **Feature Engineering**": """
        - `log_amount` — log(1 + amount), which flattens the long tail of spending.
        - `day_of_week` / `hour` — pulled off the timestamp.
        - `amount_bin` — the amount bucketed against **fixed** edges kept in
          `models/amount_bins.json`, so the same amount always lands in the same bin.
        """,
        "🔮 **Predictive Insights**": """
        **Next month:** average of the last three months.

        **Growing categories:** categories where the latest month beats the month
        three periods back.

        **Due for a revisit:** merchants where the gap since your last visit has
        reached 80% of your usual interval, measured against the newest date in
        *your file* rather than today.

        **Limits:** these are simple projections, not real forecasts.
        """,
    }
    for title, body in GLOSSARY.items():
        with st.expander(title):
            st.markdown(body)

    st.markdown("---")
    st.markdown("### 💡 Pro Tips")
    st.info("""
    **For best results:**
    1. Upload at least 3 months of history.
    2. Keep date formats consistent (ISO `YYYY-MM-DD` is safest).
    3. Include merchant names, they carry most of the categorization signal.
    4. Add `lat`/`long`/`merch_lat`/`merch_long` to turn on the geo rules.
    5. Tune the anomaly rate and risk percentile to your tolerance for false alarms.
    """)

    st.markdown("### 🛡️ Data Privacy")
    st.warning("""
    Uploaded files are processed **by the server running this app**, held in memory
    for the session and dropped when the session ends. This app writes nothing to
    disk and sends nothing anywhere else, and the models are pre-trained so they
    never learn from your data. If you're using a hosted deployment then that host
    is still a third party, so for real statements run it locally with
    `streamlit run app.py`.
    """)

st.markdown("---")
st.caption("Amount bins are pinned in `models/amount_bins.json` so results stay comparable "
           "between sessions. Regenerate them with `spending-analyzer fit-bins <csv>`.")

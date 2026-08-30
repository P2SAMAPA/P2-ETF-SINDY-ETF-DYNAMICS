import streamlit as st
import pandas as pd
import requests
import json
import glob
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="P2 SINDy ETF Dynamics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
PRIMARY = "#2563eb"      # blue
POSITIVE = "#16a34a"     # green
NEGATIVE = "#dc2626"     # red
NEUTRAL = "#d97706"      # amber
INK = "#0f172a"
SUBTLE = "#64748b"
CARD_BG = "#ffffff"
CARD_BORDER = "#e2e8f0"
PAGE_BG = "#f8fafc"

CONF_COLORS = {"high": POSITIVE, "medium": NEUTRAL, "low": NEGATIVE}

st.markdown(f"""
<style>
    .stApp {{
        background-color: {PAGE_BG};
    }}
    #MainMenu, footer {{visibility: hidden;}}

    .app-header {{
        display: flex;
        align-items: baseline;
        gap: 0.75rem;
        margin-bottom: 0;
    }}
    .app-title {{
        font-size: 1.9rem;
        font-weight: 800;
        color: {INK};
        margin: 0;
    }}
    .app-subtitle {{
        color: {SUBTLE};
        font-size: 0.95rem;
        margin-top: 0.15rem;
        margin-bottom: 1.25rem;
    }}

    .section-label {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {INK};
        margin: 1.6rem 0 0.6rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 2px solid {CARD_BORDER};
    }}

    .universe-heading {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {INK};
        margin: 0 0 0.15rem 0;
    }}
    .universe-caption {{
        color: {SUBTLE};
        font-size: 0.85rem;
        margin-bottom: 0.75rem;
    }}

    .pick-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-left: 4px solid var(--accent);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin: 0.35rem 0;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .pick-ticker {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {INK};
        letter-spacing: 0.02em;
    }}
    .pick-return {{
        font-size: 1.9rem;
        font-weight: 800;
        color: {INK};
        margin: 0.25rem 0 0.35rem 0;
        line-height: 1.1;
    }}
    .pick-badge {{
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        color: white;
        background: var(--accent);
    }}
    .pick-meta {{
        font-size: 0.75rem;
        color: {SUBTLE};
        margin-top: 0.5rem;
    }}

    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        text-align: center;
    }}
    .kpi-value {{
        font-size: 1.4rem;
        font-weight: 800;
        color: {INK};
    }}
    .kpi-label {{
        font-size: 0.72rem;
        color: {SUBTLE};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.15rem;
    }}

    .best-window-banner {{
        background: linear-gradient(90deg, #ecfdf5 0%, #f0fdf4 100%);
        border: 1px solid #bbf7d0;
        border-left: 4px solid {POSITIVE};
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-size: 0.9rem;
        color: #14532d;
        margin: 0.5rem 0 1rem 0;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid {CARD_BORDER};
        border-radius: 10px;
        overflow: hidden;
    }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_data():
    """Load latest results (local file first, then HuggingFace fallback)."""
    json_files = glob.glob("sindy_results_*.json")
    if json_files:
        latest = sorted(json_files)[-1]
        with open(latest, "r") as f:
            return json.load(f)

    try:
        repo_id = "P2SAMAPA/p2-sindy-etf-dynamics-results"
        today = datetime.now().strftime("%Y-%m-%d")
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/sindy_results_{today}.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass

    return None


def conf_color(confidence: str) -> str:
    return CONF_COLORS.get((confidence or "low").lower(), NEGATIVE)


def render_pick_cards(picks, key_prefix):
    """Render a professional row of pick cards + a horizontal bar chart."""
    if not picks:
        st.info("No ETF picks available for this selection.")
        return

    cols = st.columns(min(len(picks), 3))
    for i, pick in enumerate(picks):
        color = conf_color(pick["confidence"])
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="pick-card" style="--accent: {color};">
                <div class="pick-ticker">{pick['ticker']}</div>
                <div class="pick-return">{pick['expected_return']:+.2f}%</div>
                <span class="pick-badge">{pick['confidence']}</span>
                <div class="pick-meta">Sparsity {pick.get('sparsity', 0):.2f}
                    &nbsp;·&nbsp; {len(pick.get('active_features', []))} active term(s)</div>
            </div>
            """, unsafe_allow_html=True)

    df = pd.DataFrame(picks)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["expected_return"],
        y=df["ticker"],
        orientation="h",
        text=df["expected_return"].apply(lambda x: f"{x:+.2f}%"),
        textposition="outside",
        marker_color=[conf_color(c) for c in df["confidence"]],
        hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(160, 60 * len(picks)),
        margin=dict(l=0, r=30, t=10, b=10),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title="Expected next-day return (%)", gridcolor="#eef2f7", zeroline=True, zerolinecolor="#cbd5e1"),
        yaxis=dict(autorange="reversed"),
        font=dict(color=INK, size=12),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{key_prefix}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    data = load_data()

    st.markdown('<div class="app-header"><span style="font-size:1.9rem;">📈</span>'
                 '<span class="app-title">P2 SINDy ETF Dynamics</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Sparse Identification of Nonlinear Dynamics for ETF Selection</div>',
                unsafe_allow_html=True)

    if not data:
        st.error("No data available. Run `python trainer.py` first to generate results.")
        return

    run_date = data.get("run_date", "Unknown")
    st.caption(f"🕒 Results generated: **{run_date}**")

    tab1, tab2 = st.tabs(["📊 Top Picks", "📈 Window Backtest Results"])

    # ------------------------------------------------------------------ #
    # TAB 1 — Top Picks (best window per universe)
    # ------------------------------------------------------------------ #
    with tab1:
        top_picks = data.get("top_picks", {})
        best_window = data.get("best_window", {})

        if not top_picks:
            st.warning("No top-pick data available yet.")

        for universe, picks in top_picks.items():
            st.markdown(f'<div class="universe-heading">{universe.replace("_", " ").title()}</div>',
                        unsafe_allow_html=True)

            best = best_window.get(universe, {})
            if best:
                metrics = best.get("metrics", {})
                st.markdown(f"""
                <div class="best-window-banner">
                    ✅ Best training window: <b>{best.get('window', 'N/A')} days</b>
                    <span style="opacity:0.7;">(selected by return-prediction correlation, not Sharpe)</span>
                    &nbsp;|&nbsp; Correlation: <b>{metrics.get('correlation', 0):.4f}</b>
                    &nbsp;|&nbsp; Directional accuracy: <b>{metrics.get('directional_accuracy', 0):.1%}</b>
                    &nbsp;|&nbsp; Sharpe: <b>{metrics.get('sharpe', 0):.2f}</b>
                </div>
                """, unsafe_allow_html=True)

            render_pick_cards(picks, key_prefix=f"picks_{universe}")
            st.markdown("<div style='margin: 0.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    # TAB 2 — Window Backtest Results (+ ETF picks per window)
    # ------------------------------------------------------------------ #
    with tab2:
        backtest_results = data.get("backtest_results", {})
        window_picks = data.get("window_picks", {})

        if not backtest_results:
            st.warning("No backtest data available yet.")

        for universe, window_results in backtest_results.items():
            st.markdown(f'<div class="universe-heading">{universe.replace("_", " ").title()}</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="universe-caption">Compare performance across training window sizes</div>',
                        unsafe_allow_html=True)

            if not window_results:
                st.warning("No backtest results available for this universe.")
                continue

            df_results = pd.DataFrame([
                {
                    "Window": int(w),
                    "Correlation": r.get("correlation", 0),
                    "Directional Accuracy": r.get("directional_accuracy", 0) * 100,
                    "Sharpe Ratio": r.get("sharpe", 0),
                    "Predictions": r.get("n_predictions", 0),
                }
                for w, r in window_results.items()
            ]).sort_values("Window").reset_index(drop=True)

            best_idx = df_results["Correlation"].idxmax()
            best_row = df_results.loc[best_idx]

            # --- KPI row -------------------------------------------------
            kpi_cols = st.columns(4)
            kpi_data = [
                ("Best Window", f"{int(best_row['Window'])}d"),
                ("Correlation", f"{best_row['Correlation']:.4f}"),
                ("Directional Acc.", f"{best_row['Directional Accuracy']:.1f}%"),
                ("Sharpe (informational)", f"{best_row['Sharpe Ratio']:.2f}"),
            ]
            for col, (label, value) in zip(kpi_cols, kpi_data):
                with col:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-value">{value}</div>
                        <div class="kpi-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.caption("Best window is selected by return-prediction correlation, not Sharpe — "
                       "Sharpe reflects realized P&L, which can look good from a window whose "
                       "predictions barely explain anything.")

            st.markdown("<div style='margin-top: 0.9rem;'></div>", unsafe_allow_html=True)

            # --- Metrics table --------------------------------------------
            st.dataframe(
                df_results.style.apply(
                    lambda x: ["background-color: #dcfce7" if x.name == best_idx else "" for _ in x],
                    axis=1,
                ).format({
                    "Correlation": "{:.3f}",
                    "Directional Accuracy": "{:.1f}%",
                    "Sharpe Ratio": "{:.2f}",
                    "Predictions": "{:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Window": "Window (days)",
                },
            )

            # --- Performance chart -----------------------------------------
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_results["Window"], y=df_results["Sharpe Ratio"],
                mode="lines+markers", name="Sharpe Ratio",
                line=dict(color=PRIMARY, width=2.5), marker=dict(size=9),
            ))
            fig.add_trace(go.Scatter(
                x=df_results["Window"], y=df_results["Directional Accuracy"],
                mode="lines+markers", name="Directional Accuracy %",
                line=dict(color=POSITIVE, width=2.5, dash="dot"), marker=dict(size=9),
                yaxis="y2",
            ))
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(title="Window Size (days)", gridcolor="#eef2f7"),
                yaxis=dict(title="Sharpe Ratio", gridcolor="#eef2f7"),
                yaxis2=dict(title="Directional Accuracy (%)", overlaying="y", side="right"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                font=dict(color=INK, size=12),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"chart_backtest_{universe}")

            # --- ETF picks per window (this is what was previously missing) --
            st.markdown("###### ETF picks by window")
            universe_window_picks = window_picks.get(universe, {})

            if not universe_window_picks:
                st.info("No per-window ETF picks in this results file. Re-run `python trainer.py` "
                        "with the updated version to generate them.")
            else:
                available_windows = sorted(universe_window_picks.keys(), key=lambda w: int(w))
                window_tabs = st.tabs([f"{w}d" for w in available_windows])
                for wtab, w in zip(window_tabs, available_windows):
                    with wtab:
                        render_pick_cards(
                            universe_window_picks[w],
                            key_prefix=f"wpicks_{universe}_{w}",
                        )

            st.markdown("<hr style='margin: 1.75rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

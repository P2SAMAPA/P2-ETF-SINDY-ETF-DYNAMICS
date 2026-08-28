import streamlit as st
import pandas as pd
import requests
import json
import glob
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="P2-SINDY-ETF-DYNAMICS",
    page_icon="📈",
    layout="wide"
)

st.title("📈 P2-SINDY-ETF-DYNAMICS")
st.markdown("*Sparse Identification of Nonlinear Dynamics for ETF Selection*")


def load_data():
    """Load latest results."""
    json_files = glob.glob("sindy_results_*.json")
    if json_files:
        latest = sorted(json_files)[-1]
        with open(latest, 'r') as f:
            return json.load(f)
    
    try:
        repo_id = "P2SAMAPA/p2-sindy-etf-dynamics-results"
        today = datetime.now().strftime("%Y-%m-%d")
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/sindy_results_{today}.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    return None


def main():
    data = load_data()
    
    if not data:
        st.error("No data available. Run `python trainer.py` first.")
        return
    
    run_date = data.get('run_date', 'Unknown')
    st.caption(f"Results from: {run_date}")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Top Picks", "📈 Window Backtest Results"])
    
    with tab1:
        st.subheader("Top ETF Picks by Universe")
        
        top_picks = data.get('top_picks', {})
        
        for universe, picks in top_picks.items():
            # Show best window for this universe
            best = data.get('best_window', {}).get(universe, {})
            if best:
                st.caption(f"Best window: {best.get('window', 'N/A')} days | "
                          f"Sharpe: {best.get('metrics', {}).get('sharpe', 0):.2f}")
            
            cols = st.columns(min(len(picks), 3))
            for i, pick in enumerate(picks):
                with cols[i % len(cols)]:
                    conf = pick['confidence'].lower()
                    color = "#27ae60" if conf == "high" else "#f39c12" if conf == "medium" else "#e74c3c"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                                border-radius: 10px; padding: 1.5rem; margin: 0.5rem 0;
                                border-left: 5px solid {color};">
                        <h3 style="margin:0;">{pick['ticker']}</h3>
                        <div style="font-size:2rem; font-weight:700; margin:0.5rem 0;">
                            {pick['expected_return']:.1f}%
                        </div>
                        <div style="color:{color}; font-weight:600;">Confidence: {pick['confidence']}</div>
                        <div style="font-size:0.7rem; color:#888; margin-top:0.5rem;">
                            Sparsity: {pick.get('sparsity', 0):.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Chart
            df = pd.DataFrame(picks)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['expected_return'],
                y=df['ticker'],
                orientation='h',
                text=df['expected_return'].apply(lambda x: f"{x:.1f}%"),
                textposition='outside',
                marker_color=['#27ae60' if r > 0.5 else '#f39c12' if r > 0 else '#e74c3c' 
                              for r in df['expected_return']]
            ))
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key=f"chart_picks_{universe}")
            st.markdown("---")
    
    with tab2:
        st.subheader("Window Backtest Comparison")
        st.markdown("Compare performance of different training window sizes")
        
        backtest_results = data.get('backtest_results', {})
        
        for universe, window_results in backtest_results.items():
            st.markdown(f"### {universe}")
            
            if not window_results:
                st.warning("No backtest results available")
                continue
            
            # Create dataframe for this universe
            df_results = pd.DataFrame([
                {
                    "Window": w,
                    "Correlation": r.get("correlation", 0),
                    "Directional Accuracy": r.get("directional_accuracy", 0) * 100,
                    "Sharpe Ratio": r.get("sharpe", 0),
                    "Predictions": r.get("n_predictions", 0)
                }
                for w, r in window_results.items()
            ]).sort_values("Window")
            
            # Highlight best window
            best_idx = df_results["Sharpe Ratio"].idxmax()
            
            # Display metrics table
            st.dataframe(
                df_results.style.apply(
                    lambda x: ['background-color: #90EE90' if x.name == best_idx else '' for i in x],
                    axis=1
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Window": "Window (days)",
                    "Correlation": st.column_config.NumberColumn("Correlation", format="%.3f"),
                    "Directional Accuracy": st.column_config.NumberColumn("Directional Acc", format="%.1f%%"),
                    "Sharpe Ratio": st.column_config.NumberColumn("Sharpe Ratio", format="%.2f"),
                    "Predictions": "N Predictions"
                }
            )
            
            # Chart - Sharpe Ratio by Window
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_results["Window"],
                y=df_results["Sharpe Ratio"],
                mode='lines+markers',
                name='Sharpe Ratio',
                line=dict(color='blue', width=2),
                marker=dict(size=10)
            ))
            fig.add_trace(go.Scatter(
                x=df_results["Window"],
                y=df_results["Directional Accuracy"],
                mode='lines+markers',
                name='Directional Accuracy %',
                line=dict(color='green', width=2),
                marker=dict(size=10),
                yaxis='y2'
            ))
            fig.update_layout(
                title="Window Performance Comparison",
                xaxis_title="Window Size (days)",
                yaxis_title="Sharpe Ratio",
                yaxis2=dict(
                    title="Directional Accuracy (%)",
                    overlaying='y',
                    side='right'
                ),
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True, key=f"chart_backtest_{universe}")
            
            # Best window recommendation
            best_window = df_results.loc[best_idx]
            st.success(f"✅ **Recommended window: {int(best_window['Window'])} days** "
                      f"(Sharpe: {best_window['Sharpe Ratio']:.2f}, "
                      f"Directional: {best_window['Directional Accuracy']:.1f}%)")
            st.markdown("---")


if __name__ == "__main__":
    main()

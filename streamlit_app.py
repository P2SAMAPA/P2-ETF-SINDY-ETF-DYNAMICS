import streamlit as st
import pandas as pd
import requests
import json
import glob
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="P2-SINDY-ETF-DYNAMICS",
    page_icon="📈",
    layout="wide"
)

st.title("📈 P2-SINDY-ETF-DYNAMICS")
st.markdown("*Sparse Identification of Nonlinear Dynamics for ETF Selection*")


def load_data():
    """Load latest results."""
    # Try local files first
    json_files = glob.glob("sindy_results_*.json")
    if json_files:
        latest = sorted(json_files)[-1]
        with open(latest, 'r') as f:
            return json.load(f)
    
    # Try HuggingFace
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
    
    top_picks = data.get('top_picks', {})
    
    for universe, picks in top_picks.items():
        st.subheader(universe)
        
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
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
    
    # SINDy explanation
    with st.expander("What is SINDy?"):
        st.markdown("""
        **SINDy** discovers governing equations from data:
        
        `dX/dt = Θ(X) · Ξ`
        
        - Θ(X): library of candidate functions (polynomials, trig, etc.)
        - Ξ: sparse coefficient matrix via sparse regression
        """)


if __name__ == "__main__":
    main()

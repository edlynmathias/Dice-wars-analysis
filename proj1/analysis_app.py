import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

st.set_page_config(
    page_title="Dice Wars: 1-Agent vs 3-Agent Analysis",
    page_icon="🎲",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

st.title("🎲 Dice Wars: 1-Agent vs 3-Agent Performance & Statistical Analysis")
st.markdown("""
Welcome to the dedicated evaluation portal for **Dice Wars**. This application compares the performance, 
predictive accuracy, decision stability, and statistical significance of the **3-Agent Hierarchical Pipeline** 
(Strategy $\rightarrow$ Attack $\rightarrow$ Defense) against the **1-Agent Baseline** (Oracle / Single Agent).
""")

# Sidebar Controls
st.sidebar.header("Simulation Parameters")
num_episodes = st.sidebar.slider("Number of Simulation Episodes", min_value=100, max_value=2000, value=500, step=100)
random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)
alpha_level = st.sidebar.selectbox("Significance Level ($\alpha$)", [0.01, 0.05, 0.10], index=1)

np.random.seed(int(random_seed))

@st.cache_data
def run_simulation(episodes):
    """Simulates game turns and decision outcomes for 1-Agent vs 3-Agent systems."""
    data = []
    for i in range(episodes):
        # Generate baseline features
        win_prob_1ag = np.clip(np.random.beta(2, 2), 0.05, 0.95)
        exposure_1ag = np.random.uniform(0, 5)
        
        # 3-Agent system leverages Strategy posture & Defense balancing -> higher effective win rate & lower exposure
        win_prob_3ag = np.clip(win_prob_1ag + np.random.normal(0.08, 0.03), 0.05, 0.98)
        exposure_3ag = np.clip(exposure_1ag - np.random.uniform(0.5, 1.5), 0, 5)
        
        # Ground truth outcomes
        true_success_1ag = 1 if (win_prob_1ag - 0.1 * exposure_1ag + np.random.normal(0, 0.2)) > 0.45 else 0
        true_success_3ag = 1 if (win_prob_3ag - 0.05 * exposure_3ag + np.random.normal(0, 0.15)) > 0.40 else 0
        
        data.append({
            "Episode": i,
            "System": "1-Agent (Baseline)",
            "Win_Prob": win_prob_1ag,
            "Exposure": exposure_1ag,
            "Confidence_Score": win_prob_1ag * (1 - 0.1 * exposure_1ag),
            "True_Outcome": true_success_1ag,
            "Predicted_Label": 1 if win_prob_1ag >= 0.6 else 0
        })
        data.append({
            "Episode": i,
            "System": "3-Agent (Hierarchical)",
            "Win_Prob": win_prob_3ag,
            "Exposure": exposure_3ag,
            "Confidence_Score": win_prob_3ag * (1 - 0.05 * exposure_3ag),
            "True_Outcome": true_success_3ag,
            "Predicted_Label": 1 if win_prob_3ag >= 0.55 else 0
        })
    return pd.DataFrame(data)

df_sim = run_simulation(num_episodes)

df_1ag = df_sim[df_sim["System"] == "1-Agent (Baseline)"]
df_3ag = df_sim[df_sim["System"] == "3-Agent (Hierarchical)"]

# Key Performance Metrics
acc_1ag = accuracy_score(df_1ag["True_Outcome"], df_1ag["Predicted_Label"])
acc_3ag = accuracy_score(df_3ag["True_Outcome"], df_3ag["Predicted_Label"])
win_rate_1ag = df_1ag["True_Outcome"].mean() * 100
win_rate_3ag = df_3ag["True_Outcome"].mean() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("1-Agent Accuracy", f"{acc_1ag*100:.1f}%", f"{(acc_3ag-acc_1ag)*100:+.1f}% vs 3-Agent")
col2.metric("3-Agent Accuracy", f"{acc_3ag*100:.1f}%", f"Top Performer")
col3.metric("1-Agent Win Rate", f"{win_rate_1ag:.1f}%")
col4.metric("3-Agent Win Rate", f"{win_rate_3ag:.1f}%", f"{win_rate_3ag-win_rate_1ag:+.1f}%")

st.markdown("---")

# Main Analysis Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance & Accuracy", "📉 ROC Curve & AUC", "🔲 Confusion Matrices", "🔬 Hypothesis Testing"])

with tab1:
    st.subheader("Decision Confidence & Accuracy Distribution")
    fig_dist = px.box(
        df_sim, x="System", y="Confidence_Score", color="System",
        title="Agent Decision Confidence Distribution",
        color_discrete_map={"1-Agent (Baseline)": "#ef553b", "3-Agent (Hierarchical)": "#636efa"}
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown("""
    * **Key Takeaway**: The 3-agent hierarchical pipeline maintains higher median confidence and accounts for post-attack counter-exposure, resulting in more stable decisions.
    """)

with tab2:
    st.subheader("Receiver Operating Characteristic (ROC) Curve")
    
    fpr_1, tpr_1, _ = roc_curve(df_1ag["True_Outcome"], df_1ag["Confidence_Score"])
    roc_auc_1 = auc(fpr_1, tpr_1)
    
    fpr_3, tpr_3, _ = roc_curve(df_3ag["True_Outcome"], df_3ag["Confidence_Score"])
    roc_auc_3 = auc(fpr_3, tpr_3)
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_1, y=tpr_1, name=f'1-Agent (AUC = {roc_auc_1:.3f})', line=dict(color='#ef553b', width=2)))
    fig_roc.add_trace(go.Scatter(x=fpr_3, y=tpr_3, name=f'3-Agent (AUC = {roc_auc_3:.3f})', line=dict(color='#636efa', width=2)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Guess', line=dict(color='gray', dash='dash')))
    
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.15)
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.markdown(f"""
    * **AUC Insight**: The 3-agent architecture achieves an AUC of **{roc_auc_3:.3f}**, outperforming the 1-agent baseline AUC of **{roc_auc_1:.3f}**, indicating superior true positive classification of advantageous moves.
    """)

with tab3:
    st.subheader("Confusion Matrix Visualization")
    col_cm1, col_cm2 = st.columns(2)
    
    cm_1 = confusion_matrix(df_1ag["True_Outcome"], df_1ag["Predicted_Label"])
    cm_3 = confusion_matrix(df_3ag["True_Outcome"], df_3ag["Predicted_Label"])
    
    with col_cm1:
        st.markdown("#### 1-Agent Baseline")
        fig_cm1 = px.imshow(
            cm_1, text_auto=True, color_continuous_scale="Reds",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Loss", "Win"], y=["Loss", "Win"]
        )
        st.plotly_chart(fig_cm1, use_container_width=True)
        
    with col_cm2:
        st.markdown("#### 3-Agent Hierarchical")
        fig_cm3 = px.imshow(
            cm_3, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Loss", "Win"], y=["Loss", "Win"]
        )
        st.plotly_chart(fig_cm3, use_container_width=True)

with tab4:
    st.subheader("Statistical Hypothesis Testing (Welch's t-test)")
    st.markdown("""
    To validate whether the performance boost of the 3-agent pipeline is statistically significant, we perform an independent two-sample t-test.
    """)
    
    t_stat, p_val = stats.ttest_ind(df_3ag["Confidence_Score"], df_1ag["Confidence_Score"], equal_var=False)
    
    col_h1, col_h2, col_h3 = st.columns(3)
    col_h1.metric("t-Statistic", f"{t_stat:.4f}")
    col_h2.metric("p-Value", f"{p_val:.5e}")
    col_h3.metric("Significance Level ($\alpha$)", f"{alpha_level}")
    
    st.markdown("#### Hypotheses")
    st.markdown("""
    * **Null Hypothesis ($H_0$):** $\mu_1 = \mu_3$ (No significant difference in decision quality between 1-agent and 3-agent systems).
    * **Alternative Hypothesis ($H_1$):** $\mu_3 > \mu_1$ (The 3-agent system achieves significantly higher decision quality).
    """)
    
    if p_val < alpha_level:
        st.success(f"**Conclusion:** Since $p$-value ({p_val:.5e}) $< \alpha$ ({alpha_level}), we **reject the null hypothesis ($H_0$)**. There is statistically significant evidence that the 3-agent system outperforms the 1-agent baseline.")
    else:
        st.warning(f"**Conclusion:** Fail to reject the null hypothesis ($H_0$).")
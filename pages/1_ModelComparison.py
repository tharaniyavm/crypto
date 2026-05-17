import streamlit as pd_st
import pandas as pd

# Set page configuration for a modern wide layout
pd_st.set_page_config(
    page_title="Model Performance Analytics",
    page_icon="📊",
    layout="wide"
)

# Application Header
pd_st.title("📊 Model Evaluation Dashboard")
pd_st.markdown("Compare machine learning model metrics and instantly discover the top performer.")
pd_st.divider()

# 2. Build DataFrame based on user configuration
data = pd.read_csv("model_evaluation_comparison.csv")
df = pd.DataFrame(data)

# Find the absolute best model based on minimum MAE
best_row = df.loc[df['MAE'].idxmin()]

# 3. High-Level Summary Cards
col1, col2 = pd_st.columns(2)
with col1:
    pd_st.metric(label="🏆 Best Performing Model", value=best_row['Model'])
with col2:
    pd_st.metric(label="📉 Lowest Error (MAE)", value=f"{best_row['MAE']:.2f}")

pd_st.divider()

# 4. Styling Logic Function
def highlight_min_mae(column):
    is_min = column == column.min()
    # Hex color matching Streamlit UI aesthetics
    return ['background-color: #2e7d32; color: white; font-weight: bold' if v else '' for v in is_min]

# Apply styling parameters to the target column
styled_df = df.style.apply(highlight_min_mae, subset=['MAE']).format({
    'MAE': '{:.2f}',
    'MSE': '{:.2f}'
    
})

# 5. Render Interactive styled Table in Streamlit
pd_st.subheader("📋 Comprehensive Performance Table")
pd_st.dataframe(styled_df, use_container_width=True, hide_index=True)

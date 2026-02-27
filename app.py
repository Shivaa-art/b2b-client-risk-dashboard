# ==========================================
# B2B CLIENT RISK & CHURN DASHBOARD
# FINAL CLEAN VERSION (STREAMLIT CLOUD)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="B2B Risk Dashboard", layout="wide")

# =============================
# LOAD DATASET (EXCEL FILE)
# =============================
df = pd.read_excel("B2B_Client_Churn_5000.csv.xlsx")

# Convert Renewal_Status to numeric
df['Renewal_Status'] = df['Renewal_Status'].map({'Yes': 1, 'No': 0})

# =============================
# SIDEBAR FILTERS
# =============================
st.sidebar.header("Filters")

region_filter = st.sidebar.multiselect(
    "Select Region",
    df['Region'].unique(),
    default=df['Region'].unique()
)

industry_filter = st.sidebar.multiselect(
    "Select Industry",
    df['Industry'].unique(),
    default=df['Industry'].unique()
)

risk_filter = st.sidebar.multiselect(
    "Select Risk Category",
    df['Risk_Category'].unique(),
    default=df['Risk_Category'].unique()
)

filtered_df = df[
    (df['Region'].isin(region_filter)) &
    (df['Industry'].isin(industry_filter)) &
    (df['Risk_Category'].isin(risk_filter))
]

# =============================
# TITLE
# =============================
st.title("B2B Client Risk Intelligence Dashboard")

# =============================
# KPI SECTION
# =============================
total_clients = filtered_df.shape[0]
high_risk_clients = filtered_df[filtered_df['Risk_Category'] == 'High'].shape[0]
avg_revenue = filtered_df['Monthly_Revenue_USD'].mean()
predicted_churn_rate = (1 - filtered_df['Renewal_Status'].mean()) * 100

# Train ML model
features = [
    'Monthly_Usage_Score',
    'Payment_Delay_Days',
    'Contract_Length_Months',
    'Support_Tickets_Last30Days'
]

X = df[features]
y = df['Renewal_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Clients", total_clients)
col2.metric("High Risk Clients", high_risk_clients)
col3.metric("Predicted Churn Rate (%)", round(predicted_churn_rate, 2))
col4.metric("Average Revenue ($)", round(avg_revenue, 2))

st.markdown("---")

# =============================
# VISUALIZATIONS
# =============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Category Distribution")
    st.bar_chart(filtered_df['Risk_Category'].value_counts())

with col2:
    st.subheader("Industry-wise Risk")
    industry_risk = pd.crosstab(filtered_df['Industry'], filtered_df['Risk_Category'])
    st.bar_chart(industry_risk)

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Revenue vs Risk Score")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(
        data=filtered_df,
        x='Total Risk Score',
        y='Monthly_Revenue_USD',
        hue='Risk_Category'
    )
    st.pyplot(fig1)

with col4:
    st.subheader("Contract Length vs Renewal")
    fig2, ax2 = plt.subplots()
    sns.boxplot(
        data=filtered_df,
        x='Renewal_Status',
        y='Contract_Length_Months'
    )
    st.pyplot(fig2)

st.markdown("---")

# =============================
# TOP 20 HIGH RISK CLIENTS
# =============================
st.subheader("Top 20 High-Risk Clients")

top_high_risk = filtered_df[
    filtered_df['Risk_Category'] == 'High'
].sort_values(by='Total Risk Score', ascending=False).head(20)

st.dataframe(top_high_risk)

st.markdown("---")

# =============================
# RETENTION STRATEGY BUTTON
# =============================
if st.button("Generate Retention Strategy"):
    st.subheader("Recommended Retention Strategies")

    st.write("1️⃣ Offer payment restructuring plans for clients with high payment delays.")
    st.write("2️⃣ Assign dedicated account managers to high support-ticket clients.")
    st.write("3️⃣ Provide engagement training for low-usage customers.")
    st.write("4️⃣ Offer long-term contract discounts to short-contract clients.")
    st.write("5️⃣ Conduct quarterly relationship review meetings.")

st.markdown("---")

# =============================
# RESPONSIBLE AI SECTION
# =============================
st.subheader("Ethical Considerations")

st.write("""
- Predictive models may contain historical bias.
- Labeling clients as 'High Risk' may influence sales behavior unfairly.
- Client financial data must be handled securely.
- AI predictions should support, not replace, human decision-making.
""")

st.markdown(f"Model Accuracy (Decision Tree): **{round(accuracy*100, 2)}%**")

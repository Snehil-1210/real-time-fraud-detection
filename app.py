import streamlit as st, pandas as pd, plotly.express as px
import requests, random, time, numpy as np

st.set_page_config(page_title='Fraud Dashboard', layout='wide', page_icon='🛡️')
st.title('Real-time Fraud Detection Dashboard')

API_URL = 'http://localhost:8000/predict'

def generate_transaction():
    'Simulate a random transaction'
    return {
        'time': random.uniform(0, 172792),
        **{f'v{i}': random.gauss(0, 1) for i in range(1, 29)},
          'amount': random.uniform(0.01, 5000)}

col1, col2, col3, col4 = st.columns(4)
placeholder = st.empty()

results = []
for _ in range(20):  # Simulate 20 transactions
    tx = generate_transaction()
    try:
        r = requests.post(API_URL, json=tx, timeout=2).json()
        results.append({'Amount': round(tx['amount'],2),
                         'Fraud': r['fraud'],
                         'Risk': r['risk'],
                         'Probability': r['probability']})
    except: pass

if results:
    df = pd.DataFrame(results)
    fraud_count = df['Fraud'].sum()
    col1.metric('Transactions', len(df))
    col2.metric('Fraud Detected', int(fraud_count))
    col3.metric('Fraud Rate', f'{fraud_count/len(df)*100:.1f}%')
    col4.metric('Avg Amount', f"${df['Amount'].mean():.2f}")
    st.dataframe(df.style.apply(lambda x: ['background-color: #ffcccc' if v else '' for v in x], subset=['Fraud']))
    fig = px.scatter(df, x=df.index, y='Probability', color='Risk', title='Transaction Risk Scores')
    st.plotly_chart(fig, use_container_width=True)

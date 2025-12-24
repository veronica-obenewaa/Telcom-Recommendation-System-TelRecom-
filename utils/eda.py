import streamlit as st
import pandas as pd

def show_eda(df):
    st.header("📈 Exploratory Data Analysis")

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.subheader("Basic Statistics")
    st.dataframe(df.describe())

    if 'Churn Value' in df.columns:
        churn_rate = df['Churn Value'].mean()
        st.metric("Overall Churn Rate", f"{churn_rate:.2%}")

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

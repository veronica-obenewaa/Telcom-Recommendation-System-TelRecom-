import streamlit as st
import plotly.express as px

def show_clusters(df):
    st.header("🧠 Customer Segmentation")

    if 'Cluster' not in df.columns:
        st.warning("Clustering has not been applied yet.")
        return

    cluster_counts = df['Cluster'].value_counts().sort_index()

    fig = px.bar(
        x=cluster_counts.index.astype(str),
        y=cluster_counts.values,
        labels={"x": "Cluster", "y": "Number of Customers"},
        title="Customer Distribution by Cluster",
        color=cluster_counts.index.astype(str)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Summary")
    try:
        st.dataframe(
            df.groupby('Cluster').mean(numeric_only=True)
        )
    except Exception as e:
        st.error(f"Error displaying cluster summary: {str(e)}")

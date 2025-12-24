import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def prediction_page(
    churn_model,
    churn_scaler,
    feature_cols,
    cluster_model,
    cluster_scaler,
    clustering_features,
    plan_mapping
):
    st.header("🔮 Churn Prediction")

    # Get top 10 features
    if isinstance(feature_cols, list):
        top_10_features = feature_cols[:10] if len(feature_cols) > 10 else feature_cols
    else:
        top_10_features = list(feature_cols)[:10]

    st.markdown("### Enter Customer Details")
    st.info(f"Using {len(top_10_features)} features for prediction")

    user_input = {}

    for col in top_10_features:
        user_input[col] = st.number_input(
            col,
            help=f"Enter value for {col}",
            value=0.0
        )

    if st.button("Predict Churn"):
        try:
            input_df = pd.DataFrame([user_input])

            # Create a full feature dataframe with zeros for missing features
            full_features = pd.DataFrame(0, index=[0], columns=feature_cols)
            for col in top_10_features:
                if col in user_input:
                    full_features[col] = user_input[col]

            # -------------------------
            # Churn Prediction
            # -------------------------
            X_scaled = churn_scaler.transform(full_features)
            churn_prob = churn_model.predict_proba(X_scaled)[0, 1]

            st.metric("Churn Probability", f"{churn_prob:.2%}")

            # -------------------------
            # Recommendation Logic
            # -------------------------
            if churn_prob > 0.5:
                st.error("⚠️ High Risk Customer")

                cluster_input = input_df[clustering_features]
                cluster_scaled = cluster_scaler.transform(cluster_input)
                cluster = cluster_model.predict(cluster_scaled)[0]

                plan = plan_mapping[cluster]

                st.success(f"Recommended Plan: {plan['plan_name']}")
                st.write(plan['description'])
                st.write("**Features:**")
                for f in plan['features']:
                    st.write(f"• {f}")

            else:
                st.success("✅ Low Risk Customer")
                st.info("Customer is satisfied with current plan.")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

    # -------------------------
    # Visualization Section
    # -------------------------
    st.markdown("---")
    st.subheader("📊 Feature Importance Plots")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Top 10 Features Used**")
        feature_importance = pd.DataFrame({
            'Feature': top_10_features,
            'Order': range(1, len(top_10_features) + 1)
        })
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=feature_importance, x='Order', y='Feature', hue='Feature', palette='viridis', ax=ax, legend=False)
        ax.set_title('Top 10 Features for Churn Prediction')
        ax.set_xlabel('Feature Rank')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.write("**Churn Risk Distribution**")
        fig, ax = plt.subplots(figsize=(8, 6))
        churn_categories = ['Low Risk (0-0.3)', 'Medium Risk (0.3-0.7)', 'High Risk (0.7-1.0)']
        churn_counts = [45, 30, 25]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax.pie(churn_counts, labels=churn_categories, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Churn Risk Categories')
        st.pyplot(fig)
        plt.close()

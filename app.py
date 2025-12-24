import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import io

# Page config
st.set_page_config(
    page_title="Telco Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "mailto:support@telco-analytics.com",
        "Report a bug": "mailto:bugs@telco-analytics.com",
        "About": "# Telco Churn Prediction System\nVersion 1.0.0\nBuilt for enterprise churn analysis."
    }
)

# Custom CSS for professional styling
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #2c3e50;
    }
    [data-testid="stSidebar"] h2 {
        color: #ecf0f1;
        font-size: 20px;
        margin-bottom: 20px;
    }
    [data-testid="stRadio"] > label {
        color: #ecf0f1 !important;
    }
    [data-testid="stRadio"] > label > div {
        color: #ecf0f1 !important;
        font-size: 16px;
    }
    .stMetric {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: #b0d4ff !important;
    }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-left: 4px solid #2196F3;
        border-radius: 4px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-left: 4px solid #4caf50;
        border-radius: 4px;
    }
    .warning-box {
        background-color: #e12729;
        padding: 15px;
        border-left: 4px solid #ff9800;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

st.markdown("""
<div class="header-container">
    <h1>📊 Telco Churn Prediction System</h1>
    <p style="font-size: 16px; margin-top: 10px;">Enterprise-grade customer churn analysis & prediction platform</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    """Load telco dataset with caching"""
    try:
        return pd.read_csv("data/telco.csv")
    except FileNotFoundError:
        st.error("❌ Data file not found at 'data/telco.csv'")
        return None

@st.cache_resource
def load_models():
    """Load pre-trained models and scalers"""
    try:
        with st.spinner("🔄 Loading ML models..."):
            return {
                'churn_model': joblib.load("models/churn_model.pkl"),
                'churn_scaler': joblib.load("models/churn_scaler.pkl"),
                'kmeans': joblib.load("models/kmeans_model.pkl"),
                'cluster_scaler': joblib.load("models/cluster_scaler.pkl"),
                'feature_cols': joblib.load("config/feature_cols.pkl"),
                'clustering_features': joblib.load("config/clustering_features.pkl"),
                'plan_mapping': joblib.load("config/plan_mapping.pkl"),
            }
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

# Load data and models
df = load_data()
models = load_models()

if df is None or models is None:
    st.stop()

st.session_state.data_loaded = True

# Sidebar Navigation with Pro Features
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Select Page:",
        ["Dashboard", "Exploratory Data Analysis", "Churn Prediction", "Settings"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Refresh data button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Cache cleared! Data will refresh on next load.")
    
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("v1.0.0 | Production Ready")

# Dashboard Page
if page == "Dashboard":
    st.header("📊 Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    with col1:
        st.metric("👥 Total Customers", f"{total_customers:,}")
    
    if 'Churn Value' in df.columns:
        churned = int(df['Churn Value'].sum())
        with col2:
            st.metric("⚠️ Churned Customers", f"{churned:,}")
        
        churn_rate = df['Churn Value'].mean()
        with col3:
            st.metric("📉 Churn Rate", f"{churn_rate:.1%}")
    
    if 'Tenure Months' in df.columns:
        avg_tenure = df['Tenure Months'].mean()
        with col4:
            st.metric("📅 Avg Tenure", f"{avg_tenure:.0f} months")
    
    st.divider()
    
    st.subheader("💡 Quick Tips")
    st.markdown("""
    - Use the **Exploratory Data Analysis** page to view visualizations
    - Go to **Churn Prediction** to predict customer churn risk
    - Enter customer features and get instant predictions
    - View prediction history to track your analyses
    """)

# EDA Page
if page == "Exploratory Data Analysis":
    st.header("📈 Exploratory Data Analysis")
    
    st.divider()
    
    # Row 1: Churn Distribution & Tenure Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution")
        if 'Churn Value' in df.columns:
            churn_data = df['Churn Value'].value_counts().reset_index()
            churn_data.columns = ['Churn Status', 'Count']
            churn_data['Churn Status'] = churn_data['Churn Status'].map({0: 'No Churn', 1: 'Churn'})
            fig = px.bar(churn_data, x='Churn Status', y='Count', 
                        color='Churn Status', color_discrete_map={'No Churn': '#2ecc71', 'Churn': '#e74c3c'},
                        title='Customer Churn Distribution', text='Count')
            fig.update_traces(textposition='auto')
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Tenure Distribution")
        if 'Tenure Months' in df.columns:
            fig = px.histogram(df, x='Tenure Months', nbins=30, 
                             title='Customer Tenure Distribution',
                             labels={'Tenure Months': 'Tenure (Months)', 'count': 'Frequency'},
                             color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # Row 2: Contract Type & Internet Service
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Contract Type Distribution")
        if 'Contract' in df.columns:
            contract_data = df['Contract'].value_counts().reset_index()
            contract_data.columns = ['Contract', 'Count']
            fig = px.bar(contract_data, y='Contract', x='Count', orientation='h',
                        title='Customers by Contract Type', text='Count',
                        color_discrete_sequence=['#9b59b6'])
            fig.update_traces(textposition='auto')
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Internet Service Distribution")
        if 'Internet Service' in df.columns:
            internet_data = df['Internet Service'].value_counts().reset_index()
            internet_data.columns = ['Internet Service', 'Count']
            fig = px.pie(internet_data, values='Count', names='Internet Service',
                        title='Internet Service Type Distribution',
                        color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # Row 3: Payment Method vs Churn
    st.subheader("Payment Method vs Churn")
    if 'Payment Method' in df.columns and 'Churn Value' in df.columns:
        payment_churn = df.groupby(['Payment Method', 'Churn Value']).size().unstack(fill_value=0)
        payment_churn_reset = payment_churn.reset_index()
        payment_churn_reset.columns = ['Payment Method', 'No Churn', 'Churn']
        payment_churn_reset = payment_churn_reset.melt(id_vars='Payment Method', 
                                                        var_name='Churn Status', value_name='Count')
        fig = px.bar(payment_churn_reset, x='Payment Method', y='Count', color='Churn Status',
                    title='Churn by Payment Method', barmode='group',
                    color_discrete_map={'No Churn': '#2ecc71', 'Churn': '#e74c3c'})
        st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # Row 4: Geographic Analysis
    st.subheader("Geographic Analysis")
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Customer Locations")
            if 'Churn Value' in df.columns:
                df['Churn Label'] = df['Churn Value'].map({0: 'No Churn', 1: 'Churn'})
                fig = px.scatter(df, x='Longitude', y='Latitude', 
                               color='Churn Label',
                               title='Customer Geographic Distribution',
                               color_discrete_map={'No Churn': '#2ecc71', 'Churn': '#e74c3c'},
                               hover_data=['Churn Label'],
                               opacity=0.7)
                fig.update_traces(marker=dict(size=6))
                st.plotly_chart(fig, width='stretch')
        
        with col1:
            st.subheader("Churn Heatmap by Location")
            fig = px.density_contour(df, x='Longitude', y='Latitude',
                                   title='Churn Density by Region',
                                   nbinsx=20, nbinsy=20)
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("📍 Latitude and Longitude data not available in dataset")

# Churn Prediction Page
elif page == "Churn Prediction":
    st.header("🔮 Churn Prediction & Recommendations")
    
    with st.expander("ℹ️ How to Use", expanded=False):
        st.markdown("""
        1. **Enter Customer Details**: Fill in the top 10 key features
        2. **Click Predict**: Get churn probability and risk assessment
        3. **View Recommendations**: Receive personalized retention strategies
        4. **View History**: Track all predictions made
        """)
    
    # Feature descriptions mapping
    feature_descriptions = {
        'Dependents': 'Number of dependents (0, 1, 2, 3+)',
        'Tenure Months': 'Customer tenure in months (0-72)',
        'Monthly Charges': 'Monthly service charges ($)',
        'Total Charges': 'Total charges over lifetime ($)',
        'Contract': 'Contract type (Month-to-month, One year, Two year)',
        'Internet Service': 'Internet service type (DSL, Fiber, No)',
        'Online Security': 'Online security service (Yes/No)',
        'Tech Support': 'Tech support service (Yes/No)',
        'Payment Method': 'Payment method used (Electronic check, Mailed check, etc)',
        'Paperless Billing': 'Paperless billing enabled (Yes=1, No=0)',
    }
    
    feature_cols = list(models['feature_cols'])[:10]
    st.info(f"📌 Using top {len(feature_cols)} predictive features")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Customer Details")
        user_input = {}
        cols = st.columns(2)
        for idx, col in enumerate(feature_cols):
            with cols[idx % 2]:
                desc = feature_descriptions.get(col, "")
                if desc:
                    st.caption(f"📋 {desc}")
                user_input[col] = st.number_input(col, value=0.0, format="%.2f")
    
    with col2:
        st.subheader("Quick Presets")
        if st.button("📊 Low Risk Customer", use_container_width=True):
            st.session_state.preset = "low_risk"
        if st.button("⚠️ High Risk Customer", use_container_width=True):
            st.session_state.preset = "high_risk"
    
    st.divider()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        predict_button = st.button("🔍 Predict Churn", use_container_width=True, type="primary")
    with col2:
        st.write("")  # Spacer
    
    if predict_button:
        try:
            with st.spinner("⏳ Analyzing customer profile..."):
                full_df = pd.DataFrame(0, index=[0], columns=models['feature_cols'])
                for col in feature_cols:
                    full_df[col] = user_input[col]
                X_scaled = models['churn_scaler'].transform(full_df)
                churn_prob = models['churn_model'].predict_proba(X_scaled)[0, 1]
                
                # Store prediction in history
                prediction_record = {
                    'probability': churn_prob,
                    'timestamp': datetime.now(),
                    'input_data': user_input.copy(),
                    'risk_level': 'HIGH' if churn_prob > 0.5 else 'MEDIUM' if churn_prob > 0.3 else 'LOW'
                }
                st.session_state.prediction_history.append(prediction_record)
                st.session_state.last_prediction = prediction_record
            
            st.divider()
            st.subheader("📊 Prediction Results")
            
            st.write(f"**Churn Probability:** {churn_prob:.2%}")
            risk_level = "🔴 HIGH RISK" if churn_prob > 0.5 else "🟡 MEDIUM RISK" if churn_prob > 0.3 else "🟢 LOW RISK"
            st.write(f"**Risk Level:** {risk_level}")
            
            # Risk visualization
            fig = go.Figure(data=[
                go.Bar(x=['Churn Risk', 'Retention'], 
                      y=[churn_prob*100, (1-churn_prob)*100],
                      marker_color=['#e74c3c', '#2ecc71'],
                      text=[f'{churn_prob:.1%}', f'{(1-churn_prob):.1%}'],
                      textposition='auto')
            ])
            fig.update_layout(title="Churn vs Retention", 
                            yaxis_title="Percentage (%)",
                            showlegend=False, height=300)
            st.plotly_chart(fig, width='stretch')
            
            st.divider()
            
            # Recommendations
            if churn_prob > 0.5:
                st.markdown("""
                <div class="warning-box">
                    <h3>⚡ High Churn Risk Detected</h3>
                    <p>This customer requires immediate retention action.</p>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    input_df = pd.DataFrame([user_input])
                    cluster_scaled = models['cluster_scaler'].transform(input_df[models['clustering_features']])
                    cluster = models['kmeans'].predict(cluster_scaled)[0]
                    plan = models['plan_mapping'][cluster]
                    
                    st.subheader("💡 Recommended Retention Plan")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.success(f"**Plan Name:** {plan['plan_name']}")
                        st.write(f"**Description:** {plan['description']}")
                    with col2:
                        if st.button("📥 Apply Recommendation"):
                            st.success("✅ Recommendation applied to customer profile!")
                
                except Exception as rec_error:
                    st.warning(f"Could not generate recommendation: {str(rec_error)}")
            else:
                st.markdown("""
                <div class="success-box">
                    <h3>✅ Low Churn Risk</h3>
                    <p>This customer shows strong retention indicators.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Export functionality
            st.divider()
            st.subheader("📥 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # Create CSV download
                report_df = pd.DataFrame({
                    'Metric': ['Churn Probability', 'Risk Level', 'Timestamp'],
                    'Value': [f'{churn_prob:.2%}', 'HIGH' if churn_prob > 0.5 else 'LOW', 
                             st.session_state.last_prediction['timestamp'].strftime('%Y-%m-%d %H:%M:%S')]
                })
                csv = report_df.to_csv(index=False)
                st.download_button("📄 Download CSV Report", csv, "prediction_report.csv", "text/csv")
            
            with col2:
                # Create JSON download
                import json
                json_data = {
                    'prediction': {
                        'churn_probability': float(churn_prob),
                        'risk_level': 'HIGH' if churn_prob > 0.5 else 'LOW',
                        'timestamp': st.session_state.last_prediction['timestamp'].isoformat(),
                        'input_features': user_input
                    }
                }
                st.download_button("📋 Download JSON", json.dumps(json_data, indent=2), 
                                 "prediction_report.json", "application/json")
        
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.info("💡 Tip: Ensure all input values are valid numbers.")
    
    # Prediction History Section
    st.divider()
    st.subheader("📈 Prediction History")
    
    if st.session_state.prediction_history:
        # Create history dataframe
        history_data = []
        for i, pred in enumerate(st.session_state.prediction_history, 1):
            history_data.append({
                'Prediction #': i,
                'Churn Probability': f"{pred['probability']:.2%}",
                'Risk Level': pred['risk_level'],
                'Timestamp': pred['timestamp'].strftime('%H:%M:%S')
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        # Plot prediction trend
        st.subheader("Prediction Trend Analysis")
        probs = [p['probability'] for p in st.session_state.prediction_history]
        timestamps = [p['timestamp'].strftime('%H:%M:%S') for p in st.session_state.prediction_history]
        
        fig = px.line(
            x=list(range(len(probs))),
            y=probs,
            title='Churn Probability Trend',
            labels={'x': 'Prediction Number', 'y': 'Churn Probability'},
            markers=True,
            color_discrete_sequence=['#3498db']
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Risk Threshold")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("📭 No predictions yet. Make your first prediction to see history here.")

# Settings Page
elif page == "Settings":
    st.header("⚙️ Settings & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Application Settings")
        cache_time = st.slider("Data Cache Duration (hours)", 0, 24, 1)
        st.success(f"✅ Cache set to {cache_time} hour(s)")
        
        if st.button("🗑️ Clear All Caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ All caches cleared!")
    
    st.divider()
    
    st.subheader("📝 About This Application")
    st.markdown("""
    ### Telco Churn Prediction System v1.0.0
    
    **Purpose:**
    Enterprise-grade platform for predicting customer churn and recommending retention strategies.
    
    **Features:**
    - ✅ Real-time churn probability prediction
    - ✅ Customer risk segmentation
    - ✅ Personalized retention recommendations
    - ✅ Interactive data visualization
    - ✅ Export capabilities
    
    **Technology Stack:**
    - Python, Streamlit, Plotly
    - Scikit-learn, XGBoost, LightGBM
    - Joblib for model serialization
    
    **Support:**
    - Email: support@telco-analytics.com
    - Documentation: /docs
    - API Docs: /api/docs
    
    ---
    © 2025 Telco Analytics. All rights reserved.
    """)
    
    st.divider()
    st.caption("Built with ❤️ using Streamlit")


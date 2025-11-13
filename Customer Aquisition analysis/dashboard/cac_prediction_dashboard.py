import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  # Added for model comparison
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Define PROJECT_ROOT if not already defined (adjust path as needed)
PROJECT_ROOT = Path(__file__).parent.parent  

# Fix for import: Add src to path and try import, else fallback
src_path = PROJECT_ROOT / 'src'
if src_path.exists():
    sys.path.append(str(src_path))
    try:
        from data import load_raw  # Import directly since src is in sys.path
    except (ImportError, ModuleNotFoundError):
        load_raw = None  # Fallback
else:
    load_raw = None  # Fallback if src folder doesn't exist

# Set page config
st.set_page_config(page_title="CAC Prediction Dashboard", layout="wide", page_icon="📊")

# Step 1: Data Preparation
@st.cache_data
def load_data():
    # Use your loading line (assumes load_raw is defined/imported)
    if load_raw is not None:
        data = load_raw(str(PROJECT_ROOT / "data" / "raw" / "customer_acquisition.csv"))
    else:
        data = pd.read_csv(str(PROJECT_ROOT / "data" / "raw" / "customer_acquisition.csv"))
    
    # Debug: Show columns to verify
    st.write("Columns in your CSV:", data.columns.tolist())
    
    # Basic cleaning: Drop rows with missing critical values
    data = data.dropna(subset=['Marketing_Spend', 'Conversion_Rate', 'Revenue'])
    
    # If 'CAC' column is missing, calculate it (adjust formula if needed, e.g., CAC = cost / conversion_rate)
    if 'CAC' not in data.columns:
        data['CAC'] = data['Marketing_Spend'] / data['Conversion_Rate']  # Example; change to your logic (e.g., cost / (revenue * conversion_rate))
        st.warning("'CAC' column was missing and has been calculated as cost / conversion_rate. Adjust if incorrect.")
    
    return data

data = load_data()

# Step 2: Enhanced EDA Functions (Now using Plotly with Storytelling Annotations)
def plot_cac_distribution(data):
    fig = px.histogram(data, x='CAC', nbins=30, title='CAC Distribution: Understanding Cost Variability', 
                       labels={'CAC': 'CAC ($)'}, marginal='rug', color_discrete_sequence=['#1f77b4'])
    fig.add_annotation(text="Insight: Most CAC values cluster around the mean. Outliers may indicate inefficient campaigns.", 
                       xref="paper", yref="paper", x=0.5, y=0.9, showarrow=False)
    return fig

def plot_correlation(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, title='Correlation Matrix: Key Relationships', 
                    color_continuous_scale='RdBu_r')
    fig.add_annotation(text="Insight: Cost and CAC are positively correlated—higher spend often leads to higher CAC.", 
                       xref="paper", yref="paper", x=0.5, y=0.1, showarrow=False)
    return fig

# New: Box Plot for Variability
def plot_box_plot(data):
    fig = px.box(data, x='Marketing_Spend', y='CAC', title='Box Plot: CAC by Cost Segments', 
                 labels={'Marketing_Spend': 'Marketing_Spend ($)', 'CAC': 'CAC ($)'}, color_discrete_sequence=['#ff7f0e'])
    fig.add_annotation(text="Insight: Higher costs show wider CAC variability—optimize mid-range spends.", 
                       xref="paper", yref="paper", x=0.5, y=0.9, showarrow=False)
    return fig

# New: Pair Plot for Multivariate Insights
def plot_pair_plot(data):
    fig = px.scatter_matrix(data, dimensions=['Marketing_Spend', 'Conversion_Rate', 'Revenue', 'CAC'], 
                            title='Pair Plot: Multivariate Relationships', color='CAC', 
                            color_continuous_scale='Viridis')
    return fig



# Step 3: Model Training with Options
@st.cache_data
def train_model(data, model_type='Linear'):
    features = ['Marketing_Spend', 'Conversion_Rate', 'Revenue'] 
    
    X = data[features]
    y = data['CAC']
    
    if len(X) < 10:
        return None, None, None, None, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_type == 'Random Forest':
        model = RandomForestRegressor(random_state=42)
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(features, model.feature_importances_))
    return model, mse, r2, y_test, y_pred, feature_importance

# Step 4: Streamlit Dashboard with Storytelling
st.title("📊 Customer Acquisition Cost (CAC) Prediction Dashboard")
st.markdown("**Story Overview**: This dashboard tells the story of your customer acquisition efforts. From data exploration to predictions, discover how costs impact CAC and optimize for better ROI.")

# Progress Bar for Storytelling Flow
progress = st.progress(0)
st.markdown("### Step 1: Data Overview")
progress.progress(20)

# Filters for Interactivity
st.sidebar.header("Filters for Exploration")
cost_range = st.sidebar.slider("Filter by Cost Range ($)", int(data['Marketing_Spend'].min()), int(data['Marketing_Spend'].max()), (int(data['Marketing_Spend'].min()), int(data['Marketing_Spend'].max())))
filtered_data = data[(data['Marketing_Spend'] >= cost_range[0]) & (data['Marketing_Spend'] <= cost_range[1])]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average CAC", f"${filtered_data['CAC'].mean():.2f}")
with col2:
    st.metric("Total Cost", f"${filtered_data['Marketing_Spend'].sum():,.0f}")
with col3:
    st.metric("Conversion Rate Avg", f"{filtered_data['Conversion_Rate'].mean():.2%}")

st.markdown("**Key Takeaway**: Filtering by cost reveals how higher spends correlate with CAC—use this to target efficient campaigns.")

st.header("Exploratory Data Analysis")
st.markdown("### Step 2: Deep Dive EDA")
progress.progress(50)

col4, col5 = st.columns(2)
with col4:
    fig_dist = plot_cac_distribution(filtered_data)
    st.plotly_chart(fig_dist)
with col5:
    fig_corr = plot_correlation(filtered_data)
    st.plotly_chart(fig_corr)

# New: Additional EDA Plots
col6, col7 = st.columns(2)
with col6:
    st.plotly_chart(plot_box_plot(filtered_data))
with col7:
    st.plotly_chart(plot_pair_plot(filtered_data))



st.markdown("**Story Insight**: EDA shows CAC variability—box plots highlight outliers, and correlations guide feature selection for predictions.")

# Model Evaluation with Enhancements
st.markdown("### Step 3: Model Insights")
progress.progress(75)

model_type = st.selectbox("Choose Model for Prediction", ['Linear', 'Random Forest'])
model, mse, r2, y_test, y_pred, feature_importance = train_model(filtered_data, model_type)

if model and mse and r2:
    st.write(f"**Model Performance**: MSE = {mse:.2f}, R² = {r2:.2f}")
    if r2 > 0.7:
        st.success("Model performs well—predictions are reliable!")
    else:
        st.warning("Model is moderate; consider more data or features.")
    
    # Actual vs. Predicted Plot (Plotly)
    fig_scatter = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual CAC ($)', 'y': 'Predicted CAC ($)'}, title='Actual vs. Predicted CAC')
    fig_scatter.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Perfect Fit', line=dict(dash='dash', color='red')))
    st.plotly_chart(fig_scatter)
    
    # New: Feature Importance (if available)
    if feature_importance:
        fig_importance = px.bar(x=list(feature_importance.keys()), y=list(feature_importance.values()), 
                                title='Feature Importance: What Drives CAC?', labels={'x': 'Feature', 'y': 'Importance'})
        st.plotly_chart(fig_importance)
        st.markdown("**Insight**: Conversion rate is the top driver—focus on improving it to lower CAC.")
else:
    st.error("Model could not be trained (e.g., insufficient data).")

# Interactive Plotly Chart
st.header("Interactive CAC Scatter Plot")
fig = px.scatter(filtered_data, x='Marketing_Spend', y='CAC', color='Conversion_Rate', title="Marketing_Spend vs. CAC")
st.plotly_chart(fig)

# Predictions with Storytelling
st.markdown("### Step 4: What-If Predictions")
progress.progress(90)

st.sidebar.header("Predict CAC for New Data")
future_cost = st.sidebar.number_input("Marketing-Spend ($)", min_value=0.0, value=25000.0)
future_conversion = st.sidebar.slider("Conversion Rate (0-1)", 0.0, 1.0, 0.5)
future_revenue = st.sidebar.number_input("Revenue ($)", min_value=0.0, value=50000.0)

if model:
    predicted_cac = model.predict([[future_cost, future_conversion, future_revenue]])[0]
    st.metric("Predicted CAC", f"${predicted_cac:.2f}")
    st.markdown("**Recommendation**: If predicted CAC > $X, reduce cost or boost conversion to stay profitable.")
    
    # New: Export Option
    if st.button("Export Predictions to CSV"):
        predictions_df = pd.DataFrame({'Marketing_Spend': [future_cost], 'Conversion_Rate': [future_conversion], 
                                       'Revenue': [future_revenue], 'Predicted_CAC': [predicted_cac]})
        predictions_df.to_csv('predictions.csv', index=False)
        st.success("Predictions exported to predictions.csv!")

st.markdown("### Conclusion: The CAC Story")
progress.progress(100)
st.markdown("**Final Takeaway**: By analyzing costs and conversions, we've uncovered patterns to optimize CAC. Use predictions to simulate scenarios and drive better acquisition strategies. ROI improves when CAC < revenue per customer!")

# Data Table
st.header("Data")
st.dataframe(filtered_data)
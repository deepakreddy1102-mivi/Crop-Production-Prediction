
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ===================== PAGE CONFIGURATION =====================
st.set_page_config(
    page_title="Crop Production Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== LOAD MODEL AND ARTIFACTS =====================
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/random_forest_tuned.pkl')
    le_area = joblib.load('models/le_area.pkl')
    le_item = joblib.load('models/le_item.pkl')
    area_list = joblib.load('models/area_list.pkl')
    item_list = joblib.load('models/item_list.pkl')
    year_list = joblib.load('models/year_list.pkl')
    area_harvested_stats = joblib.load('models/area_harvested_stats.pkl')
    yield_stats = joblib.load('models/yield_stats.pkl')
    return model, le_area, le_item, area_list, item_list, year_list, area_harvested_stats, yield_stats

model, le_area, le_item, area_list, item_list, year_list, area_harvested_stats, yield_stats = load_artifacts()

# Load cleaned data for visualizations
@st.cache_data
def load_data():
    return pd.read_csv('data/FAOSTAT_cleaned.csv')

df = load_data()

# ===================== SIDEBAR =====================
st.sidebar.title("🌾 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA Dashboard", "🔮 Predict Production", "📈 Model Performance"])

# ===================== HOME PAGE =====================
if page == "🏠 Home":
    st.title("🌾 Crop Production Prediction System")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌍 Countries", f"{df['Area'].nunique()}")
    col2.metric("🌱 Crops", f"{df['Item'].nunique()}")
    col3.metric("📅 Years", f"{df['Year'].min()} - {df['Year'].max()}")
    col4.metric("📦 Total Records", f"{len(df):,}")

    st.markdown("---")
    st.subheader("About This Project")
    st.write("""
    This application predicts **crop production (in tonnes)** based on agricultural factors 
    such as area harvested, yield, crop type, country, and year. The prediction model is a 
    **Tuned Random Forest Regressor** trained on FAOSTAT data (2019-2023) covering 200 countries 
    and 157 crop types.
    """)

    st.subheader("How to Use")
    st.write("""
    - **📊 EDA Dashboard:** Explore data trends, top crops, top countries, and more
    - **🔮 Predict Production:** Select inputs from dropdowns and get instant production forecasts
    - **📈 Model Performance:** View model comparison metrics and evaluation results
    """)

# ===================== EDA DASHBOARD =====================
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")
    st.markdown("---")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        selected_countries = st.multiselect("Select Countries", area_list, default=["India", "China", "United States of America"])
    with col2:
        selected_year = st.selectbox("Select Year", sorted(df['Year'].unique(), reverse=True))

    st.markdown("---")

    # Top 10 Crops by Production for selected year
    st.subheader(f"Top 10 Crops by Production ({selected_year})")
    year_data = df[df['Year'] == selected_year]
    top_crops = year_data.groupby('Item')['Production'].sum().nlargest(10).reset_index()
    fig1 = px.bar(top_crops, x='Production', y='Item', orientation='h',
                  color='Production', color_continuous_scale='Greens',
                  labels={'Production': 'Production (tonnes)', 'Item': 'Crop'})
    fig1.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)
    st.plotly_chart(fig1, use_container_width=True)

    # Production Comparison across selected countries
    if selected_countries:
        st.subheader(f"Production Comparison — Selected Countries ({selected_year})")
        country_data = year_data[year_data['Area'].isin(selected_countries)]
        country_prod = country_data.groupby('Area')['Production'].sum().reset_index()
        fig2 = px.bar(country_prod, x='Area', y='Production', color='Area',
                      labels={'Production': 'Total Production (tonnes)', 'Area': 'Country'})
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Yearly Production Trends for selected countries
    if selected_countries:
        st.subheader("Yearly Production Trends — Selected Countries")
        trend_data = df[df['Area'].isin(selected_countries)].groupby(
            ['Year', 'Area'])['Production'].sum().reset_index()
        fig3 = px.line(trend_data, x='Year', y='Production', color='Area',
                       markers=True, labels={'Production': 'Total Production (tonnes)'})
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    # Yield vs Area Harvested Scatter
    st.subheader(f"Yield vs Area Harvested ({selected_year})")
    fig4 = px.scatter(year_data, x='Area_harvested', y='Yield', color='Item',
                      hover_data=['Area', 'Production'],
                      labels={'Area_harvested': 'Area Harvested (ha)', 'Yield': 'Yield (kg/ha)'},
                      log_x=True, log_y=True)
    fig4.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

# ===================== PREDICT PRODUCTION =====================
elif page == "🔮 Predict Production":
    st.title("🔮 Crop Production Prediction")
    st.markdown("---")
    st.write("Select the inputs below to predict crop production (in tonnes):")

    col1, col2 = st.columns(2)

    with col1:
        selected_area = st.selectbox("🌍 Select Country", area_list, index=area_list.index("India"))
        selected_item = st.selectbox("🌱 Select Crop", item_list, index=item_list.index("Rice"))
        selected_year_pred = st.selectbox("📅 Select Year", list(range(2019, 2026)))

    with col2:
        area_harvested = st.slider(
            "🏞️ Area Harvested (hectares)",
            min_value=1,
            max_value=int(area_harvested_stats['max']),
            value=int(area_harvested_stats['median']),
            step=100
        )
        yield_val = st.slider(
            "📊 Yield (kg/ha)",
            min_value=1,
            max_value=int(yield_stats['max']),
            value=int(yield_stats['median']),
            step=100
        )

    st.markdown("---")

    if st.button("🚀 Predict Production", use_container_width=True):
        try:
            area_encoded = le_area.transform([selected_area])[0]
            item_encoded = le_item.transform([selected_item])[0]

            input_data = pd.DataFrame({
                'Area_encoded': [area_encoded],
                'Item_encoded': [item_encoded],
                'Year': [selected_year_pred],
                'Area_harvested': [area_harvested],
                'Yield': [yield_val]
            })

            prediction = model.predict(input_data)[0]

            st.success(f"### Predicted Production: **{prediction:,.2f} tonnes**")

            # Show input summary
            st.subheader("Input Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric("Country", selected_area)
            summary_col2.metric("Crop", selected_item)
            summary_col3.metric("Year", selected_year_pred)

            summary_col4, summary_col5, summary_col6 = st.columns(3)
            summary_col4.metric("Area Harvested", f"{area_harvested:,} ha")
            summary_col5.metric("Yield", f"{yield_val:,} kg/ha")
            summary_col6.metric("Predicted Production", f"{prediction:,.0f} tonnes")

            # Show historical data for same crop and country
            st.subheader(f"Historical Data: {selected_item} in {selected_area}")
            hist_data = df[(df['Area'] == selected_area) & (df['Item'] == selected_item)]
            if not hist_data.empty:
                fig = px.line(hist_data, x='Year', y='Production', markers=True,
                              labels={'Production': 'Production (tonnes)'})
                fig.add_scatter(x=[selected_year_pred], y=[prediction],
                               mode='markers', name='Predicted',
                               marker=dict(size=15, color='red', symbol='star'))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical data available for this crop-country combination.")

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# ===================== MODEL PERFORMANCE =====================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance Comparison")
    st.markdown("---")

    # Load model comparison results
    results_df = pd.read_csv('outputs/model_comparison.csv')

    # Display metrics table
    st.subheader("Model Comparison Table")
    st.dataframe(results_df.style.highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='lightgreen')
                 .highlight_max(subset=['R2_Test'], color='lightgreen'),
                 use_container_width=True)

    # R2 comparison chart
    st.subheader("R² Score Comparison")
    fig_r2 = go.Figure(data=[
        go.Bar(name='Train R²', x=results_df['Model'], y=results_df['R2_Train'], marker_color='#3498db'),
        go.Bar(name='Test R²', x=results_df['Model'], y=results_df['R2_Test'], marker_color='#e74c3c')
    ])
    fig_r2.update_layout(barmode='group', height=450, yaxis_range=[0, 1.1])
    st.plotly_chart(fig_r2, use_container_width=True)

    # MAE comparison chart
    st.subheader("MAE Comparison (Lower is Better)")
    fig_mae = px.bar(results_df, x='Model', y='MAE', color='Model',
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig_mae.update_layout(height=400)
    st.plotly_chart(fig_mae, use_container_width=True)

    # Best model summary
    st.subheader("🏆 Best Model: Random Forest (Tuned)")
    best_col1, best_col2, best_col3, best_col4 = st.columns(4)
    best_row = results_df[results_df['Model'] == 'Random Forest (Tuned)'].iloc[0]
    best_col1.metric("R² (Test)", f"{best_row['R2_Test']}")
    best_col2.metric("MAE", f"{best_row['MAE']:,.0f}")
    best_col3.metric("RMSE", f"{best_row['RMSE']:,.0f}")
    best_col4.metric("MSE", f"{best_row['MSE']:,.0f}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Built with** Streamlit + Scikit-learn")
st.sidebar.markdown("**Data Source:** FAOSTAT (2019-2023)")

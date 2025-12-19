# ERCOT Grid Strain Dashboard – June 2025 (Enhanced)
# Author: Bareethul Kader
# Enhanced with modern UI, interactive charts, and advanced features

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="ERCOT Grid Strain Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
    <style>
    /* Main background and text */
    .main {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    /* Metric cards styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #00d4ff;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0a0c0;
        font-size: 0.9rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1a2e;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #16213e;
        color: #a0a0c0;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0f3460;
        color: #00d4ff;
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(90deg, #00d4ff 0%, #0080ff 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    csv_path = "grid_strain_enriched.csv"
    df = pd.read_csv(csv_path, parse_dates=["Datetime"])
    df["Temperature_C"] = pd.to_numeric(df["Temperature_C"], errors="coerce")
    df["Temperature_C"] = df["Temperature_C"].replace(999.9, pd.NA)
    df["Temperature_C"] = df["Temperature_C"].infer_objects(copy=False).interpolate(method="linear", limit_direction="both")
    df = df[df["Datetime"].dt.date != pd.to_datetime("2025-06-22").date()]
    df["Hour"] = df["Datetime"].dt.hour
    df["Weekday"] = df["Datetime"].dt.day_name()
    df = df[df["Temperature_C"] != 999.0]
    threshold = df["Total_Demand_MWh"].quantile(0.90)
    df["Grid_Strain_Flag"] = df["Total_Demand_MWh"] > threshold
    return df

# Header with icon
st.markdown("# ⚡ ERCOT Grid Strain Dashboard")
st.markdown("### Real-time analysis of demand, generation mix, temperature, and CO₂ emissions")

# Load data
with st.spinner("Loading data..."):
    df = load_data()

# Sidebar Filters
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.markdown("---")
    
    # Date filter
    st.markdown("### 📅 Date Filter")
    available_days = sorted(df["Datetime"].dt.date.unique())
    selected_day_option = st.selectbox(
        "Select Day:",
        options=["All Days"] + [str(day) for day in available_days],
        help="Choose a specific day or view all data"
    )
    
    # Hour range
    st.markdown("### 🕐 Hour Range")
    hour_range = st.slider(
        "Select Hours:",
        0, 23, (0, 23),
        help="Filter data by hour of day"
    )
    
    st.markdown("---")
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        show_raw = st.checkbox("Show Raw Data Table")
        strain_threshold = st.slider("Grid Strain Threshold (%)", 85, 95, 90, 1)
    
    st.markdown("---")
    st.markdown("### 📊 Data Summary")

# Filter data
if selected_day_option == "All Days":
    filtered_df = df[(df["Hour"] >= hour_range[0]) & (df["Hour"] <= hour_range[1])].copy()
else:
    selected_day = pd.to_datetime(selected_day_option).date()
    filtered_df = df[(df["Datetime"].dt.date == selected_day) &
                     (df["Hour"] >= hour_range[0]) &
                     (df["Hour"] <= hour_range[1])].copy()

# Recalculate strain with custom threshold
if not filtered_df.empty:
    threshold = filtered_df["Total_Demand_MWh"].quantile(strain_threshold / 100)
    filtered_df["Grid_Strain_Flag"] = filtered_df["Total_Demand_MWh"] > threshold
else:
    threshold = np.nan



# If filters return no rows, stop early to avoid downstream plotting errors
if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filters. Try widening the hour range or selecting 'All Days'.")
    st.stop()


# Sidebar stats
with st.sidebar:
    st.metric("Total Records", f"{len(filtered_df):,}")
    st.metric("Date Range", f"{len(available_days)} days")
    st.metric("Strain Threshold", "—" if pd.isna(threshold) else f"{threshold:,.0f} MWh")

# KPI Metrics Row
st.markdown("## 📈 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    peak_demand = filtered_df["Total_Demand_MWh"].max()
    avg_demand = filtered_df["Total_Demand_MWh"].mean()
    delta = ((peak_demand - avg_demand) / avg_demand * 100)
    st.metric("🔺 Peak Demand", f"{peak_demand:,.0f} MWh", f"+{delta:.1f}% vs avg")

with col2:
    max_temp = filtered_df["Temperature_C"].max()
    avg_temp = filtered_df["Temperature_C"].mean()
    st.metric("🌡️ Max Temperature", f"{max_temp:.1f}°C", f"{avg_temp:.1f}°C avg")

with col3:
    avg_co2 = filtered_df["CO2_Intensity_ton_per_MWh"].mean()
    st.metric("♻️ CO₂ Intensity", f"{avg_co2:.3f} t/MWh")

with col4:
    strain_pct = (filtered_df["Grid_Strain_Flag"].sum() / len(filtered_df)) * 100
    strain_hours = filtered_df["Grid_Strain_Flag"].sum()
    st.metric("⚠️ Grid Strain", f"{strain_pct:.1f}%", f"{strain_hours} hours")

with col5:
    gen_cols = ["Solar Generation (MWh)", "Wind Generation (MWh)", "Natural Gas Generation (MWh)",
                "Coal Generation (MWh)", "Nuclear Generation (MWh)", "Battery storage Generation (MWh)"]
    renewable_share = (filtered_df["Solar Generation (MWh)"] + filtered_df["Wind Generation (MWh)"]).sum() / filtered_df[gen_cols].sum().sum() * 100
    st.metric("🌱 Renewable Share", f"{renewable_share:.1f}%")

# Show raw data if requested
if show_raw:
    st.markdown("### 📋 Raw Data")
    st.dataframe(filtered_df, use_container_width=True, height=300)

# Tabs Layout
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Demand Analysis",
    "🌡️ Temperature Impact",
    "⚡ Generation Mix",
    "🌍 Emissions",
    "📥 Export & Insights"
])

# Tab 1: Demand Analysis
with tab1:
    st.markdown("## 📊 Demand Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Interactive demand over time
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=filtered_df["Datetime"],
            y=filtered_df["Total_Demand_MWh"],
            mode='lines',
            name='Demand',
            line=dict(color='#00d4ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.1)'
        ))
        
        # Add strain threshold line
        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                     annotation_text="Strain Threshold")
        
        fig.update_layout(
            title="Total Demand Over Time",
            xaxis_title="Time",
            yaxis_title="Demand (MWh)",
            template="plotly_dark",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Demand distribution
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=filtered_df["Total_Demand_MWh"],
            nbinsx=30,
            marker_color='#00d4ff',
            name='Demand Distribution'
        ))
        
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                     annotation_text="Threshold")
        
        fig.update_layout(
            title="Demand Distribution",
            xaxis_title="Demand (MWh)",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Hourly pattern
    st.markdown("### ⏰ Hourly Demand Pattern")
    
    hourly_avg = filtered_df.groupby("Hour").agg({
        "Total_Demand_MWh": ["mean", "min", "max"]
    }).reset_index()
    hourly_avg.columns = ["Hour", "Mean", "Min", "Max"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hourly_avg["Hour"],
        y=hourly_avg["Max"],
        fill=None,
        mode='lines',
        line_color='rgba(255, 100, 100, 0.3)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly_avg["Hour"],
        y=hourly_avg["Min"],
        fill='tonexty',
        mode='lines',
        line_color='rgba(255, 100, 100, 0.3)',
        name='Range',
        fillcolor='rgba(255, 100, 100, 0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly_avg["Hour"],
        y=hourly_avg["Mean"],
        mode='lines+markers',
        name='Average Demand',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=8)
    ))
    
    # Highlight peak hours
    fig.add_vrect(x0=15, x1=19, fillcolor="orange", opacity=0.1,
                  annotation_text="Peak Hours", annotation_position="top left")
    
    fig.update_layout(
        title="Average Hourly Demand Pattern (with Min/Max Range)",
        xaxis_title="Hour of Day",
        yaxis_title="Demand (MWh)",
        template="plotly_dark",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Weekday comparison
    st.markdown("### 📅 Demand by Weekday")
    
    df_no_sunday = filtered_df[filtered_df["Weekday"] != "Sunday"]
    
    fig = go.Figure()
    
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
        day_data = df_no_sunday[df_no_sunday["Weekday"] == day]["Total_Demand_MWh"]
        fig.add_trace(go.Box(y=day_data, name=day, marker_color='#00d4ff'))
    
    fig.update_layout(
        title="Demand Distribution by Weekday",
        yaxis_title="Demand (MWh)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Temperature Impact
with tab2:
    st.markdown("## 🌡️ Temperature Impact Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature vs Demand scatter
        fig = px.scatter(
            filtered_df,
            x="Temperature_C",
            y="Total_Demand_MWh",
            color="Hour",
            color_continuous_scale="Turbo",
            title="Temperature vs Demand (Colored by Hour)",
            labels={"Temperature_C": "Temperature (°C)", "Total_Demand_MWh": "Demand (MWh)"},
            template="plotly_dark",
            height=400
        )
        
        # Add trendline
        
# Prepare data for trendline
trend_df = filtered_df.dropna(subset=["Temperature_C", "Total_Demand_MWh"])

if len(trend_df) >= 3:  # safety check for polynomial fit
    z = np.polyfit(trend_df["Temperature_C"], trend_df["Total_Demand_MWh"], 2)
    p = np.poly1d(z)

    x_trend = np.linspace(
        trend_df["Temperature_C"].min(),
        trend_df["Temperature_C"].max(),
        100
    )

    fig.add_trace(
        go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode="lines",
            name="Polynomial Fit",
            line=dict(color="red", width=2, dash="dash")
        )
    )

# Render chart
st.plotly_chart(fig, use_container_width=True)

    with col2:
    # Temperature comparison: strain vs normal
    strain_temps = filtered_df[filtered_df["Grid_Strain_Flag"]]["Temperature_C"]
    normal_temps = filtered_df[~filtered_df["Grid_Strain_Flag"]]["Temperature_C"]

    fig = go.Figure()

    fig.add_trace(go.Box(
        y=normal_temps,
        name="Normal",
        marker_color="#4c4c4c",
        boxmean="sd"
    ))

    fig.add_trace(go.Box(
        y=strain_temps,
        name="Grid Strain",
        marker_color="#ff6b6b",
        boxmean="sd"
    ))

    fig.update_layout(
        title="Temperature During Grid Strain vs Normal",
        yaxis_title="Temperature (°C)",
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Dual axis: temp and demand by hour
    st.markdown("### ⏰ Hourly Temperature and Demand Patterns")
    
    hourly_stats = filtered_df.groupby("Hour").agg({
        "Temperature_C": "mean",
        "Total_Demand_MWh": "mean"
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=hourly_stats["Hour"], y=hourly_stats["Temperature_C"],
                  name="Temperature", line=dict(color='#ff6b6b', width=3),
                  mode='lines+markers'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=hourly_stats["Hour"], y=hourly_stats["Total_Demand_MWh"],
                  name="Demand", line=dict(color='#00d4ff', width=3),
                  mode='lines+markers'),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Hour of Day")
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False, color='#ff6b6b')
    fig.update_yaxes(title_text="Demand (MWh)", secondary_y=True, color='#00d4ff')
    
    fig.update_layout(
        title="Average Hourly Temperature vs Demand",
        template="plotly_dark",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Temperature bins
    st.markdown("### 🌡️ Demand by Temperature Range")
    
    bins = [0, 27, 30, 32, 50]
    labels = ["<27°C", "27-30°C", "30-32°C", ">32°C"]
    filtered_df_temp = filtered_df.dropna(subset=["Temperature_C"]).copy()
    filtered_df_temp["Temp_Bin"] = pd.cut(filtered_df_temp["Temperature_C"], bins=bins, labels=labels)
    
    temp_demand = filtered_df_temp.groupby("Temp_Bin", observed=True)["Total_Demand_MWh"].mean().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=temp_demand["Temp_Bin"],
        y=temp_demand["Total_Demand_MWh"],
        marker=dict(
            color=temp_demand["Total_Demand_MWh"],
            colorscale='YlOrRd',
            showscale=True,
            colorbar=dict(title="Demand (MWh)")
        ),
        text=temp_demand["Total_Demand_MWh"].apply(lambda x: f'{x:,.0f}'),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Average Demand by Temperature Bucket",
        xaxis_title="Temperature Range",
        yaxis_title="Average Demand (MWh)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Generation Mix
with tab3:
    st.markdown("## ⚡ Generation Mix Analysis")
    
    gen_cols = ["Solar Generation (MWh)", "Wind Generation (MWh)", "Natural Gas Generation (MWh)",
                "Coal Generation (MWh)", "Nuclear Generation (MWh)", "Battery storage Generation (MWh)"]
    
    # Hourly generation stack
    st.markdown("### 📊 Hourly Generation Mix")
    
    gen_hourly = filtered_df.groupby("Hour")[gen_cols].mean().reset_index()
    
    fig = go.Figure()
    
    colors = {
        "Solar Generation (MWh)": "#FFD700",
        "Wind Generation (MWh)": "#00CED1",
        "Natural Gas Generation (MWh)": "#FF6347",
        "Coal Generation (MWh)": "#696969",
        "Nuclear Generation (MWh)": "#9370DB",
        "Battery storage Generation (MWh)": "#32CD32"
    }
    
    for col in gen_cols:
        fig.add_trace(go.Scatter(
            x=gen_hourly["Hour"],
            y=gen_hourly[col],
            name=col.replace(" Generation (MWh)", ""),
            mode='lines',
            stackgroup='one',
            fillcolor=colors[col],
            line=dict(width=0.5, color=colors[col])
        ))
    
    fig.update_layout(
        title="Average Generation Mix by Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Generation (MWh)",
        template="plotly_dark",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fuel share pie chart
        st.markdown("### 🥧 Overall Fuel Mix")
        
        fuel_totals = filtered_df[gen_cols].sum()
        fuel_totals.index = [col.replace(" Generation (MWh)", "") for col in fuel_totals.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=fuel_totals.index,
            values=fuel_totals.values,
            hole=0.4,
            marker=dict(colors=[colors[col] for col in gen_cols])
        )])
        
        fig.update_layout(
            title="Total Generation Share by Fuel Type",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Strain vs normal comparison
        st.markdown("### ⚠️ Generation During Grid Strain")
        
        fuel_share = (
            filtered_df.groupby("Grid_Strain_Flag")[gen_cols].mean()
            .div(filtered_df.groupby("Grid_Strain_Flag")[gen_cols].mean().sum(axis=1), axis=0)
            * 100
        ).T
        
        fuel_share = fuel_share.reindex(columns=[False, True], fill_value=0)
        fuel_share.columns = ["Normal", "Strain"]
        fuel_share.index = [col.replace(" Generation (MWh)", "") for col in fuel_share.index]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Normal',
            x=fuel_share.index,
            y=fuel_share["Normal"],
            marker_color='#4c4c4c'
        ))
        
        fig.add_trace(go.Bar(
            name='Grid Strain',
            x=fuel_share.index,
            y=fuel_share["Strain"],
            marker_color='#ff6b6b'
        ))
        
        fig.update_layout(
            title="Fuel Share: Normal vs Strain Conditions",
            xaxis_title="Fuel Type",
            yaxis_title="Share (%)",
            barmode='group',
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Renewable vs fossil
    st.markdown("### 🌱 Renewable vs Fossil Fuel Generation")
    
    filtered_df["Renewable"] = filtered_df["Solar Generation (MWh)"] + filtered_df["Wind Generation (MWh)"]
    filtered_df["Fossil"] = filtered_df["Natural Gas Generation (MWh)"] + filtered_df["Coal Generation (MWh)"]
    
    renewable_hourly = filtered_df.groupby("Hour")[["Renewable", "Fossil", "Nuclear Generation (MWh)"]].mean().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=renewable_hourly["Hour"],
        y=renewable_hourly["Renewable"],
        name="Renewable",
        fill='tozeroy',
        line=dict(color='#32CD32', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=renewable_hourly["Hour"],
        y=renewable_hourly["Fossil"],
        name="Fossil Fuel",
        fill='tozeroy',
        line=dict(color='#FF6347', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=renewable_hourly["Hour"],
        y=renewable_hourly["Nuclear Generation (MWh)"],
        name="Nuclear",
        fill='tozeroy',
        line=dict(color='#9370DB', width=2)
    ))
    
    fig.update_layout(
        title="Renewable vs Fossil Fuel Generation by Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Generation (MWh)",
        template="plotly_dark",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Emissions
with tab4:
    st.markdown("## 🌍 CO₂ Emissions Analysis")
    
    # Emissions over time
    st.markdown("### 📈 Emissions and Intensity Trends")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=filtered_df["Datetime"], y=filtered_df["CO2_Emissions_tons"],
                  name="CO₂ Emissions", line=dict(color='#ff6b6b', width=2),
                  fill='tozeroy', fillcolor='rgba(255, 107, 107, 0.1)'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=filtered_df["Datetime"], y=filtered_df["CO2_Intensity_ton_per_MWh"],
                  name="CO₂ Intensity", line=dict(color='#00d4ff', width=2)),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="CO₂ Emissions (tons)", secondary_y=False, color='#ff6b6b')
    fig.update_yaxes(title_text="CO₂ Intensity (tons/MWh)", secondary_y=True, color='#00d4ff')
    
    fig.update_layout(
        title="CO₂ Emissions and Intensity Over Time",
        template="plotly_dark",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
# Tab 5: Export & Insights
with tab5:
    st.markdown("## 📥 Export Data & Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        # Emissions during strain
        fig = go.Figure()
        
        strain_emissions = filtered_df[filtered_df["Grid_Strain_Flag"]]["CO2_Emissions_tons"]
        normal_emissions = filtered_df[~filtered_df["Grid_Strain_Flag"]]["CO2_Emissions_tons"]
        
        fig.add_trace(go.Box(
            y=normal_emissions,
            name="Normal",
            marker_color='#4c4c4c',
            boxmean='sd'
        ))
        
        fig.add_trace(go.Box(
            y=strain_emissions,
            name="Grid Strain",
            marker_color='#ff6b6b',
            boxmean='sd'
        ))
        
        fig.update_layout(
            title="CO₂ Emissions: Normal vs Grid Strain",
            yaxis_title="CO₂ Emissions (tons)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Emissions by hour
        hourly_emissions = filtered_df.groupby("Hour")["CO2_Emissions_tons"].mean().reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=hourly_emissions["Hour"],
            y=hourly_emissions["CO2_Emissions_tons"],
            marker=dict(
                color=hourly_emissions["CO2_Emissions_tons"],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="CO₂ (tons)")
            )
        ))
        
        fig.update_layout(
            title="Average Hourly CO₂ Emissions",
            xaxis_title="Hour of Day",
            yaxis_title="CO₂ Emissions (tons)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("### 🔗 Correlation Analysis")
    
    corr_cols = ["Total_Demand_MWh", "Solar Generation (MWh)", "Wind Generation (MWh)",
                 "Natural Gas Generation (MWh)", "Coal Generation (MWh)", "Nuclear Generation (MWh)",
                 "CO2_Emissions_tons", "CO2_Intensity_ton_per_MWh", "Temperature_C"]
    
    df_corr = filtered_df[corr_cols].dropna()
    corr_matrix = df_corr.corr()
    
    # Shorten labels for better display
    labels = ["Demand", "Solar", "Wind", "Gas", "Coal", "Nuclear", "CO₂ Emissions", "CO₂ Intensity", "Temp"]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Correlation Matrix: Key Variables",
        template="plotly_dark",
        height=600,
        width=800
    )
    st.plotly_chart(fig, use_container_width=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 💾 Download Options")
    
    # Download filtered data
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📊 Download Filtered Data (CSV)",
        data=csv,
        file_name=f"ercot_data_{selected_day_option}.csv",
        mime="text/csv"
    )
    
    # Download summary statistics
    summary_stats = filtered_df.describe().T
    summary_csv = summary_stats.to_csv()
    st.download_button(
        label="📈 Download Summary Statistics",
        data=summary_csv,
        file_name=f"ercot_summary_{selected_day_option}.csv",
        mime="text/csv"
    )

with col2:
    st.markdown("### 🎯 Quick Insights")
    
    # Calculate key insights
    peak_hour = filtered_df.loc[filtered_df["Total_Demand_MWh"].idxmax(), "Hour"]
    peak_temp = filtered_df["Temperature_C"].max()
    renewable_pct = (filtered_df["Solar Generation (MWh)"] + filtered_df["Wind Generation (MWh)"]).sum() / filtered_df[gen_cols].sum().sum() * 100
    
    st.info(f"""
    **Peak Demand Hour:** {int(peak_hour)}:00
    
    **Highest Temperature:** {peak_temp:.1f}°C
    
    **Renewable Share:** {renewable_pct:.1f}%
    
    **Grid Strain Events:** {filtered_df['Grid_Strain_Flag'].sum()} hours
    """)

# Automated insights
st.markdown("### 🤖 Automated Analysis")

insights = []

# Insight 1: Peak demand correlation with temperature
temp_corr = filtered_df["Temperature_C"].corr(filtered_df["Total_Demand_MWh"])
if temp_corr > 0.7:
    insights.append(f"🌡️ **Strong positive correlation** ({temp_corr:.2f}) between temperature and demand - cooling load is a major driver")

# Insight 2: Renewable performance
if renewable_pct > 30:
    insights.append(f"🌱 **High renewable generation** at {renewable_pct:.1f}% - excellent clean energy performance")
else:
    insights.append(f"⚠️ **Low renewable generation** at {renewable_pct:.1f}% - opportunity for clean energy expansion")

# Insight 3: Grid strain timing
strain_hours = filtered_df[filtered_df["Grid_Strain_Flag"]]["Hour"].mode()
if len(strain_hours) > 0:
    insights.append(f"⚠️ **Grid strain most common** around {int(strain_hours[0])}:00 - plan demand response programs")

# Insight 4: Emissions intensity
avg_intensity = filtered_df["CO2_Intensity_ton_per_MWh"].mean()
if avg_intensity > 0.5:
    insights.append(f"🌍 **High carbon intensity** at {avg_intensity:.3f} tons/MWh - consider cleaner generation mix")
else:
    insights.append(f"✅ **Relatively clean grid** at {avg_intensity:.3f} tons/MWh carbon intensity")

for insight in insights:
    st.markdown(f"- {insight}")

# Recommendations
st.markdown("### 💡 Recommendations")

st.success("""
**Based on the current analysis:**

1. **Peak Hours (15:00-19:00):** Deploy demand response programs and energy storage during these critical hours

2. **Temperature Threshold:** When temperature exceeds 30°C, expect significant demand increases - pre-position resources

3. **Renewable Integration:** Solar peaks align well with demand - consider expanding solar capacity

4. **Grid Strain Prevention:** Focus on the top 10% demand hours - these drive grid stability concerns

5. **Emissions Reduction:** Natural gas is the marginal fuel during strain - battery storage can help reduce emissions
""")

# Comparative analysis
st.markdown("### 📊 Period Comparison")

if selected_day_option != "All Days":
    # Compare selected day to all days average
    all_days_avg = df["Total_Demand_MWh"].mean()
    selected_day_avg = filtered_df["Total_Demand_MWh"].mean()
    difference = ((selected_day_avg - all_days_avg) / all_days_avg) * 100
    
    comparison_df = pd.DataFrame({
        "Metric": ["Avg Demand", "Peak Demand", "Avg Temperature", "CO₂ Intensity"],
        "Selected Day": [
            f"{selected_day_avg:,.0f} MWh",
            f"{filtered_df['Total_Demand_MWh'].max():,.0f} MWh",
            f"{filtered_df['Temperature_C'].mean():.1f}°C",
            f"{filtered_df['CO2_Intensity_ton_per_MWh'].mean():.3f} t/MWh"
        ],
        "All Days Avg": [
            f"{all_days_avg:,.0f} MWh",
            f"{df['Total_Demand_MWh'].max():,.0f} MWh",
            f"{df['Temperature_C'].mean():.1f}°C",
            f"{df['CO2_Intensity_ton_per_MWh'].mean():.3f} t/MWh"
        ],
        "Difference": [
            f"{difference:+.1f}%",
            f"{((filtered_df['Total_Demand_MWh'].max() - df['Total_Demand_MWh'].max()) / df['Total_Demand_MWh'].max() * 100):+.1f}%",
            f"{((filtered_df['Temperature_C'].mean() - df['Temperature_C'].mean()) / df['Temperature_C'].mean() * 100):+.1f}%",
            f"{((filtered_df['CO2_Intensity_ton_per_MWh'].mean() - df['CO2_Intensity_ton_per_MWh'].mean()) / df['CO2_Intensity_ton_per_MWh'].mean() * 100):+.1f}%"
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

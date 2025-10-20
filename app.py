# ERCOT Grid Strain Dashboard – June 2025
# Author: Bareethul Kader
# Purpose: Interactive Streamlit dashboard for analyzing ERCOT demand,
#          generation mix, temperature, and CO₂ emissions during June 2025.

#Import required libraries
import streamlit as st             # Streamlit for interactive web app
import pandas as pd                # Pandas for data handling
import matplotlib.pyplot as plt    # Matplotlib for plotting
import seaborn as sns              # Seaborn for nicer plots
import numpy as np                 # NumPy for math operations
import os                          # OS for file paths

# Page Configuration
st.set_page_config(page_title="ERCOT Grid Strain Dashboard", layout="wide")
plt.rcParams["font.family"] = "DejaVu Sans" #Consistent font
sns.set_style("darkgrid")                   #Clean plot background
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

st.markdown("""
    <style>
    /* Existing styles... */

    body {
        background-color: #111;
        color: white;
    }

    /* Fix for metric truncation */
    [data-testid="stMetricValue"] {
        white-space: normal;
        font-weight: bold;
        font-size: 1rem;
        color: white;
    }

    [data-testid="stMetricLabel"] {
        white-space: normal;
        color: #cccccc;
    }
    </style>
""", unsafe_allow_html=True)

st.title("ERCOT Grid Strain Dashboard")
st.markdown("An interactive dashboard to explore demand, generation mix, temperature, and CO₂ emissions.")

# Load Data
@st.cache_data
def load_data():
    """
        Loads ERCOT dataset, cleans invalid values, and creates useful features.
        Cached so it runs only once unless the file changes.
        """
    csv_path = "grid_strain_enriched.csv" # Path to dataset
    df = pd.read_csv(csv_path, parse_dates=["Datetime"]) # Load CSV into DataFrame, parse Datetime column into proper datetime objects
    df["Temperature_C"] = pd.to_numeric(df["Temperature_C"], errors="coerce")   # Convert Temperature to numeric (any errors become NaN)
    df["Temperature_C"] = df["Temperature_C"].replace(999.9, pd.NA) # Replace placeholder 999.9 with NaN (represents missing sensor readings)
    df["Temperature_C"] = df["Temperature_C"].infer_objects(copy=False).interpolate(method="linear", limit_direction="both")  # Fill missing values in Temperature by linear interpolation
    return df

df = load_data()
df = df[df["Datetime"].dt.date != pd.to_datetime("2025-06-22").date()] # Exclude June 22, 2025 due to incomplete/missing data to ensure consistency in analysis
df["Hour"] = df["Datetime"].dt.hour # Extract useful time based features
df["Weekday"] = df["Datetime"].dt.day_name()
df = df[df["Temperature_C"] != 999.0]
threshold = df["Total_Demand_MWh"].quantile(0.90) # Defining threshold for "Grid Strain" = top 10% of demand values
df["Grid_Strain_Flag"] = df["Total_Demand_MWh"] > threshold # Create Boolean flag column (True if demand above threshold)

# Sidebar Filters
st.sidebar.header("📅 Filters")

# Dropdown: to choose a specific day or analyze across all days
available_days = sorted(df["Datetime"].dt.date.unique())
selected_day_option = st.sidebar.selectbox("Select Day:", options=["All Days"] + [str(day) for day in available_days])

hour_range = st.sidebar.slider("Select Hour Range", 0, 23, (0, 23)) # Slider: select hour range (e.g., 0–23 means full day)

# Filter dataset based on user input
if selected_day_option == "All Days":
    filtered_df = df[(df["Hour"] >= hour_range[0]) & (df["Hour"] <= hour_range[1])]
else:
    selected_day = pd.to_datetime(selected_day_option).date()
    filtered_df = df[(df["Datetime"].dt.date == selected_day) &
                     (df["Hour"] >= hour_range[0]) &
                     (df["Hour"] <= hour_range[1])]
# Calculate Grid Strain Flag
if not filtered_df.empty:
    threshold = filtered_df["Total_Demand_MWh"].quantile(0.90)
    filtered_df = filtered_df.copy()  # Avoid SettingWithCopyWarning
    filtered_df["Grid_Strain_Flag"] = filtered_df["Total_Demand_MWh"] > threshold
else:
    threshold = np.nan
    filtered_df["Grid_Strain_Flag"] = False

# KPI Metrics
st.markdown("### Key Metrics")

col1, col2, col3, col4 = st.columns(4)

# KPI 1: Peak demand in filtered range
with col1:
    peak_demand = filtered_df["Total_Demand_MWh"].max()
    st.metric("🔺 Peak Demand", f"{peak_demand:,.0f} MWh")

# KPI 2: Maximum temperature in filtered range
with col2:
    max_temp = filtered_df["Temperature_C"].max()
    st.metric("🌡️ Max Temperature", f"{max_temp:.1f} °C")

# KPI 3: Average CO₂ intensity in filtered range
with col3:
    avg_co2_intensity = filtered_df["CO2_Intensity_ton_per_MWh"].mean()
    st.metric("♻️ Avg CO₂ Intensity", f"{avg_co2_intensity:.2f} tons/MWh")

# KPI 4: % of hours under grid strain condition
with col4:
    strain_pct = (filtered_df["Grid_Strain_Flag"].sum() / len(filtered_df)) * 100
    st.metric(" ⚠️ Grid Strain Hours", f"{strain_pct:.1f}%")

# Show strain hour count in sidebar
st.sidebar.write("🔺 Grid Strain Threshold:", threshold)
st.sidebar.write("⚠️ Strain Hours on Selected Day:", filtered_df['Grid_Strain_Flag'].sum())
if st.sidebar.checkbox("Show Raw Data"):
    st.dataframe(filtered_df)

# Tabs Layout
tab1, tab2, tab3, tab4 = st.tabs([" Demand", " Temperature", " Generation", " Emissions"])

# Tab 1: Demand Analysis
with tab1:
    st.subheader("Total Demand Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered_df["Datetime"], filtered_df["Total_Demand_MWh"], label="Total Demand (MWh)")
    ax.set_xlabel("Time")
    ax.set_ylabel("System Demand (MWh)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Average Hourly Demand Pattern")
    hourly_avg = filtered_df.groupby(filtered_df["Hour"])["Total_Demand_MWh"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    hourly_avg.plot(kind='line', marker='o', ax=ax, color="teal")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Demand (MWh)")
    ax.axvspan(15, 19, color='orange', alpha=0.2, label='Typical Peak Hours')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Demand Distribution by Weekday")
    filtered_df.loc[:, "Weekday (June 16–21 2025)"] = filtered_df["Datetime"].dt.day_name()
    df_no_sunday = filtered_df[filtered_df["Weekday"] != "Sunday"]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df_no_sunday, x="Weekday", y="Total_Demand_MWh",
                order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}, ax=ax)
    ax.set_ylabel("Total Demand (MWh)")
    st.pyplot(fig)

    st.subheader("Demand Histogram")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(filtered_df["Total_Demand_MWh"], bins=20, color="skyblue", edgecolor="black")
    ax.set_xlabel("Total Demand (MWh)")
    ax.set_ylabel("Frequency")
    threshold = df["Total_Demand_MWh"].quantile(0.90)
    ax.axvline(threshold, color="red", linestyle="--", label="Grid Strain Threshold")
    ax.legend()
    st.pyplot(fig)

# Tab 2: Temperature
with tab2:
    # Scatter plot: temperature vs demand
    st.subheader("Temperature vs Total Demand")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=filtered_df, x="Temperature_C", y="Total_Demand_MWh", alpha=0.6, color="teal", ax=ax)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Total Demand (MWh)")
    scatter = plt.scatter(
        filtered_df["Temperature_C"], filtered_df["Total_Demand_MWh"],
        c=filtered_df["Hour"], cmap="coolwarm", alpha=0.7
    )
    plt.colorbar(scatter, label="Hour of Day")
    st.pyplot(fig)

    st.subheader("Temperature (°C) During Grid Strain vs Normal")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=filtered_df,
                x=filtered_df["Grid_Strain_Flag"].map({False: "Normal", True: "Strain"}),
                y="Temperature_C",
                palette=["#4c4c4c", "#c44e52"], ax=ax)
    ax.set_xlabel("Grid Strain (Top 10% Demand)")
    st.pyplot(fig)

    # Dual-axis plot: hourly avg temperature vs demand
    st.subheader("Average Hourly Temperature vs Demand")

    hourly_stats = filtered_df.groupby("Hour").agg(
        avg_temp=("Temperature_C", "mean"),
        avg_demand=("Total_Demand_MWh", "mean")
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot Temperature
    ax.plot(hourly_stats.index, hourly_stats["avg_temp"], color="red", label="Temperature (°C)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Temperature (°C)", color="red")

    # Create second y-axis for Demand
    ax2 = ax.twinx()
    ax2.plot(hourly_stats.index, hourly_stats["avg_demand"], color="blue", label="Demand (MWh)")
    ax2.set_ylabel("Demand (MWh)", color="blue")

    # Add combined legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")

    st.pyplot(fig)

    st.subheader("Average Demand by Temperature Bucket")
    bins = [0, 27, 30, 32, 50]
    labels = ["<27°C", "27–30°C", "30–32°C", ">32°C"]
    filtered_df = filtered_df.dropna(subset=["Temperature_C"])
    filtered_df.loc[:, "Temp Bin"] = pd.cut(filtered_df["Temperature_C"], bins=bins, labels=labels)
    temp_demand = filtered_df.groupby("Temp Bin")["Total_Demand_MWh"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=temp_demand, x="Temp Bin", y="Total_Demand_MWh", hue="Temp Bin",
                palette="YlOrRd", dodge=False, legend=False, ax=ax)
    for i, v in enumerate(temp_demand["Total_Demand_MWh"]):
        ax.text(i, v + 500, f"{v:,.0f}", ha='center', color='black', fontweight='bold')
    st.pyplot(fig)

    st.subheader("Temperature vs Demand (Linear vs Polynomial Trendline)")
    df_poly = filtered_df.dropna(subset=["Temperature_C", "Total_Demand_MWh"])
    # Ensure columns are numeric and drop NaNs
    df_poly["Temperature_C"] = pd.to_numeric(df_poly["Temperature_C"], errors="coerce")
    df_poly["Total_Demand_MWh"] = pd.to_numeric(df_poly["Total_Demand_MWh"], errors="coerce")
    df_poly = df_poly.dropna(subset=["Temperature_C", "Total_Demand_MWh"])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df_poly, x="Temperature_C", y="Total_Demand_MWh", color="blue", alpha=0.5, ax=ax)
    linear_coeffs = np.polyfit(df_poly["Temperature_C"], df_poly["Total_Demand_MWh"], deg=1)
    x_smooth = np.linspace(df_poly["Temperature_C"].min(), df_poly["Temperature_C"].max(), 200)
    ax.plot(x_smooth, np.poly1d(linear_coeffs)(x_smooth), color="green", linestyle="--", label="Linear")
    poly_coeffs = np.polyfit(df_poly["Temperature_C"], df_poly["Total_Demand_MWh"], deg=2)
    ax.plot(x_smooth, np.poly1d(poly_coeffs)(x_smooth), color="red", label="Polynomial")
    ax.legend()
    st.pyplot(fig)

# Tab 3: Generation
with tab3:
        st.subheader("Generation Mix by Hour – June 2025")
        # Define generation source columns
        gen_cols = ["Solar Generation (MWh)", "Wind Generation (MWh)", "Natural Gas Generation (MWh)",
                    "Coal Generation (MWh)", "Nuclear Generation (MWh)", "Battery storage Generation (MWh)"]

        gen_hourly = filtered_df.groupby("Hour")[gen_cols].mean()  # ✅ Use filtered_df
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.stackplot(gen_hourly.index, gen_hourly.T, labels=gen_cols, alpha=0.8)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Generation (MWh)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        st.pyplot(fig)

        st.subheader("Fuel Share by Demand Quartile")
        filtered_df["Demand_Quartile"] = pd.qcut(filtered_df["Total_Demand_MWh"], 4,
                                                 labels=["Q1", "Q2", "Q3", "Q4"])  # ✅
        gen_quartile = filtered_df.groupby("Demand_Quartile")[gen_cols].mean()
        gen_quartile_percent = gen_quartile.div(gen_quartile.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(8, 6))
        gen_quartile_percent.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_xlabel("Demand Quartile")
        ax.set_ylabel("Fuel Share (%)")
        st.pyplot(fig)

        st.subheader("Fuel Share During Grid Strain vs Normal")

        # Calculate average generation share for strain vs normal
        fuel_share = (
                filtered_df.groupby("Grid_Strain_Flag")[gen_cols].mean()
                .div(filtered_df.groupby("Grid_Strain_Flag")[gen_cols].mean().sum(axis=1), axis=0)
                * 100
        ).T

        # Force both categories to exist (False = Normal, True = Strain)
        fuel_share = fuel_share.reindex(columns=[False, True], fill_value=0)
        fuel_share.columns = ["Normal", "Strain"]

        # Plot side-by-side bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        fuel_share.plot(kind="bar", ax=ax)
        ax.set_xlabel("Fuel Type")
        ax.set_ylabel("Share (%)")
        ax.set_title("Fuel Share During Grid Strain vs Normal")
        st.pyplot(fig)

# Tab 4: Emissions
with tab4:
    # Dual-axis line chart: emissions vs intensity
    st.subheader("CO2 Emissions and Intensity Over Time")

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Plot CO2 Emissions
    ax1.plot(filtered_df["Datetime"], filtered_df["CO2_Emissions_tons"], color="red", label="CO2 Emissions (tons)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("CO2 Emissions (tons)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    # Add second y-axis for CO2 Intensity
    ax2 = ax1.twinx()
    ax2.plot(filtered_df["Datetime"], filtered_df["CO2_Intensity_ton_per_MWh"], color="blue",
             label="CO2 Intensity (tons/MWh)")
    ax2.set_ylabel("CO2 Intensity (tons/MWh)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Titles and layout
    fig.suptitle("CO2 Emissions and Intensity Over Time")
    fig.tight_layout()

    st.pyplot(fig)

    st.subheader("CO2 Emissions During Grid Strain vs Normal")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=filtered_df,
                x=filtered_df["Grid_Strain_Flag"].map({False: "Normal", True: "Strain"}),
                y="CO2_Emissions_tons",
                palette=["#4c4c4c", "#c44e52"], ax=ax)
    ax.set_xlabel("Grid Strain (Top 10% Demand)")
    ax.set_ylabel("CO2 Emissions (tons)")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr_cols = ["Total_Demand_MWh", "Solar Generation (MWh)", "Wind Generation (MWh)",
                 "Natural Gas Generation (MWh)", "Coal Generation (MWh)", "Nuclear Generation (MWh)",
                 "CO2_Emissions_tons", "CO2_Intensity_ton_per_MWh", "Temperature_C"]

    df_corr = filtered_df[corr_cols].dropna()
    corr_matrix = df_corr.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar_kws={"shrink": 0.8})
    st.pyplot(fig)

# End of ERCOT Grid Strain Dashboard



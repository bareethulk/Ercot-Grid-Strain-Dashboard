# ERCOT Grid Strain Dashboard

## Overview
This project analyzes ERCOT June 2025 grid strain conditions using system demand, generation mix, temperature, and CO₂ emissions data.  
An interactive Streamlit dashboard was developed to explore grid stress, fuel dependence, and emissions intensity during periods of high demand.

## Features
- Dynamic filtering by **date** and **hour range**.
- Key Metrics: Peak demand, max temperature, grid strain hours, avg CO₂ intensity.
- Visualizations:
  - Demand: hourly patterns, histograms, time trends.
  - Temperature vs Demand: scatter, hourly overlays, regression lines, temperature bins.
  - Generation: hourly fuel mix, strain vs normal fuel share.
  - Emissions: time trends, correlation heatmap.

## Dataset
- ERCOT system demand and generation data (June 2025).
- NOAA temperature data.
- EPA CAMPD emissions (2024 proxy for CO₂).

## Live demo
https://ercot-grid-strain-dashboard-jbk.streamlit.app/ 
## Run Locally
Clone this repo and install dependencies:
```bash
git clone https://github.com/<your-username>/ercot-grid-strain-dashboard.git
cd ercot-grid-strain-dashboard
pip install -r requirements.txt
streamlit run app.py


# NYC Taxi Trip Dashboard 🚖

An interactive dashboard built using Dash and Plotly to explore NYC taxi trip data. This app allows users to filter and analyze taxi trip data based on vendor, fare amount, and pickup date range.

## 📊 Features

- **Summary Table**: Displays key statistics such as total trips, average fare, most common payment method, and most frequent pickup location.
- **Histogram**: Visualizes the distribution of fare amounts by vendor.
- **Scatter Plot**: Shows the relationship between trip distance and duration, colored by payment type.
- **Map**: Displays pickup locations on an interactive map, colored by vendor.
- **Interactive Filters**: Dropdown for vendor selection, fare amount slider, and date range picker.

## 🗃 Dataset

The app uses a cleaned NYC Taxi dataset: `nyc_taxi_data_cleaned.csv`. It must include at least the following columns:

- `VendorID`
- `fare_amount`
- `tpep_pickup_datetime`
- `pickup_latitude`
- `pickup_longitude`
- `trip_distance`
- `trip_duration`
- `payment_type`

Make sure the file is available in the working directory or adjust the file path accordingly in the script.

## 🚀 Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt

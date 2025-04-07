import pandas as pd
import plotly.express as px
import numpy as np
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import dash_table

df = pd.read_csv("C:/Users/Acer/Downloads/archive (3)/train_2015.csv")

df = df[df['fare_amount'] > 0]

def remove_rows_latitude_longitude_not_in_range(df):
    MAX_LONGITUDE = -72.586532
    MIN_LONGITUDE = -74.663242

    MAX_LATITUDE = 41.959555
    MIN_LATITUDE = 40.168973,

    df = df[(MIN_LONGITUDE <= df.dropoff_longitude) &(df.dropoff_longitude <= MAX_LONGITUDE)
            & (MIN_LONGITUDE <= df.pickup_longitude) &(df.pickup_longitude <= MAX_LONGITUDE)
            & (MIN_LATITUDE <= df.dropoff_latitude) & (df.dropoff_latitude <= MAX_LATITUDE)
            & (MIN_LATITUDE <= df.pickup_latitude) & (df.pickup_latitude <= MAX_LATITUDE)]

    return df

df = remove_rows_latitude_longitude_not_in_range(df)

df = df[(df['passenger_count'] > 0) & (df['passenger_count'] < 7)]

df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

def calculate_outlier_percentage(df):
    outlier_percentages = {}

    for col in df.select_dtypes(include=['number']).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        count_upper = (df[col] > upper_bound).sum()
        count_lower = (df[col] < lower_bound).sum()
        outlier_percentage = round((count_upper + count_lower) / df.shape[0] * 100, 2)

        outlier_percentages[col] = outlier_percentage

    return outlier_percentages

outlier_percentages = calculate_outlier_percentage(df)

def drop_outliers(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' is not numeric.")

    df_clean = df.copy()

    q1 = df_clean[column].quantile(0.25)
    q3 = df_clean[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]

    return df_clean

def remove_outliers_below_threshold(df, threshold=5):
    outlier_percentages = calculate_outlier_percentage(df)
    for col, percentage in outlier_percentages.items():
        if percentage < threshold:
            df = drop_outliers(df, col)
    return df

df = remove_outliers_below_threshold(df)

print(f'Total number of Taxi trips: {df.shape[0]}')

unique_vendors = df['VendorID'].unique()
print(f'Unique taxi vendors: {unique_vendors}')

print(f'Total number of trips by each vendor: \n{df["VendorID"].value_counts()}')

"""# **🚶‍♂️ Building the Dash App**"""

vendor_dropdown = dcc.Dropdown(
            id='vendor-dropdown',
            options=[{'label': str(vendor), 'value': vendor} for vendor in unique_vendors],
            value=list(unique_vendors),
            multi=True,
            style={'width': '50%', 'margin': 'auto'})

fare_slider = dcc.Slider(
            id='fare-slider',
            min=df['fare_amount'].min(),
            max=df['fare_amount'].max(),
            step=1,
            value=df['fare_amount'].median(),
            marks={int(i): str(int(i)) for i in range(int(df['fare_amount'].min()), int(df['fare_amount'].max()) + 1, 50)},
            tooltip={"placement": "bottom", "always_visible": True}
        )

date_picker = dcc.DatePickerRange(
            id='date-picker',
            min_date_allowed=df['tpep_pickup_datetime'].min().date(),
            max_date_allowed=df['tpep_pickup_datetime'].max().date(),
            start_date=df['tpep_pickup_datetime'].min().date(),
            end_date=df['tpep_pickup_datetime'].max().date()
)

def create_table(dataframe) :
        return dash_table.DataTable(
        id='table',
        columns=[{'name': col, 'id': col} for col in dataframe.columns],
        data=dataframe.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'backgroundColor': 'rgb(240,240,240)', 'fontWeight': 'bold'},
        page_size=10
    )

def scatter_map_plot(dataframe, lat_col='pickup_latitude', lon_col='pickup_longitude', color_col='VendorID', title='Pickup Locations'):
    fig = px.scatter_mapbox(
        dataframe,
        lat=lat_col,
        lon=lon_col,
        color=color_col,
        zoom=10,
        mapbox_style='open-street-map',
        title=title
    )
    return dcc.Graph(figure=fig)


def histogram(dataframe,fare='fare_amount',VendorID='VendorID',title='Fare Amount Distribution by Vendor'):
    fig = px.histogram(
        dataframe,
        x=fare,
        color=VendorID,
        nbins=50,
        title=title
    )
    return dcc.Graph(figure=fig)

def scatter_plot(dataframe,tripdist='trip_distance',trip_duration = 'trip_duration',payment = 'payment_type',title='Trip Duration vs Distance'):
    fig = px.scatter(
        dataframe,
        x=tripdist,
        y=trip_duration,
        color=payment,
        title=title
    )
    return dcc.Graph(figure=fig)

def get_summary_stats(df):
    total_trips = df.shape[0]
    average_fare = round(df['fare_amount'].mean(), 2)
    most_common_payment = df['payment_type'].mode()[0]

    pickup_location_series = df['pickup_latitude'].astype(str) + ', ' + df['pickup_longitude'].astype(str)
    most_common_pickup_location = pickup_location_series.mode()[0]

    summary_df = pd.DataFrame({
        'Metric': ['Total Trips', 'Average Fare', 'Most Common Payment', 'Most Frequent Pickup Location'],
        'Value': [total_trips, average_fare, most_common_payment, most_common_pickup_location]
    })

    return summary_df

df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
df = df[df['trip_duration'] > 0]

df['pickup_time_str'] = df['tpep_pickup_datetime'].dt.strftime('%Y-%m-%d %H:%M')
df['dropoff_time_str'] = df['tpep_dropoff_datetime'].dt.strftime('%Y-%m-%d %H:%M')

unique_vendors = df['VendorID'].dropna().unique()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.server = server

app.layout = dbc.Container([
    html.H1("NYC Taxi Trip Dashboard", className="text-center my-4"),

    dbc.Row([
        dbc.Col(vendor_dropdown, md=4),
        dbc.Col(fare_slider, md=4),
        dbc.Col(date_picker, md=4)
    ], className="mb-4"),

    dbc.Tabs([
        dbc.Tab(label="Summary", tab_id="tab-summary"),
        dbc.Tab(label="Fare Histogram", tab_id="tab-histogram"),
        dbc.Tab(label="Trip Duration vs Distance", tab_id="tab-scatter"),
        dbc.Tab(label="Pickup Locations", tab_id="tab-map")
    ], id="tabs", active_tab="tab-summary"),

    html.Div(id="tab-content", className="mt-4")
])

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    Input("vendor-dropdown", "value"),
    Input("fare-slider", "value"),
    Input("date-picker", "start_date"),
    Input("date-picker", "end_date")
)
def update_content(tab, selected_vendors, fare_limit, start_date, end_date):
    dff = df[
        (df['VendorID'].isin(selected_vendors)) &
        (df['fare_amount'] <= fare_limit) &
        (df['tpep_pickup_datetime'].dt.date >= pd.to_datetime(start_date).date()) &
        (df['tpep_pickup_datetime'].dt.date <= pd.to_datetime(end_date).date())
    ]

    if tab == "tab-summary":
        return create_table(get_summary_stats(dff))

    elif tab == "tab-histogram":
        return histogram(dff)

    elif tab == "tab-scatter":
        return scatter_plot(dff)

    elif tab == "tab-map":
        return scatter_map_plot(dff)

    return html.Div("No content available")

if __name__ == "__main__":
    app.run(debug=True)
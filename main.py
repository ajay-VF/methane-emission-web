from datetime import datetime
import os
import pandas as pd
import requests
import zipfile
import joblib
from joblib import load
from io import BytesIO
import ee
import numpy as np
import rasterio
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

json_data = st.secrets["json_data"]
service_account = st.secrets["service_account"]

# Authorising the app
credentials = ee.ServiceAccountCredentials(service_account, key_data=json_object)
ee.Initialize(credentials)
img = Image.open('data/Vasudha_Logo_PNG.png')
# Resize the image
resized_img = img.resize((300, 300))  # Adjust the width and height as needed

# Create two columns
col1, col2 = st.columns([1, 5])

# Display the resized image in the first column
col1.image(resized_img, use_column_width=True)

# Display the title in the second column
col2.title('Methane Emission Calculation Web App')
district_data = pd.read_csv('data/District_corrected_names.csv')
states = district_data['STATE'].unique().tolist()  # Get unique states
st.write('Select state and district')
col1, col2 = st.columns(2)
with col1:
    state_dropdown = st.selectbox('State:', states)
with col2:
    districts = district_data[district_data['STATE'] == state_dropdown]['District_1'].tolist()
    district_dropdown = st.selectbox('District:', districts)
page = st.sidebar.selectbox("Select a page", ["Methane for Selected Date", "Monthly Methane for Selected Year", "How to use"])



district_data1=district_data[district_data['STATE'] == state_dropdown]
district_id=district_data1[district_data1['District_1'] == district_dropdown]['DISTRICT_L'].values[0]
# Google Earth Engine initialization and function to load shapefiles
ee.Initialize()
def data_landsat8_download(districts, selected_date,district_id,state ):

    ee.Initialize()
    # Load the shapefile for the given district from Earth Engine assets
    def load_district_shapefile(district_id):
        return ee.FeatureCollection("projects/ee-my-vikas-2413/assets/DISTRICTs_Corrected").filter(ee.Filter.equals('DISTRICT_L', str(district_id)))
      
    polygon = load_district_shapefile(district_id)
    # Visualization parameters
    # Parse the date
    date = datetime.strptime(selected_date, '%d-%m-%Y')
    start = ee.Date(date)
    end = start.advance(28, 'day')
    # Get the first feature from the FeatureCollection
    first_feature = polygon.first()
    # Extract the geometry from the feature
    geometry = first_feature.geometry()
    ############ remove below if error occurs
    # Flatten the list of coordinates
    coordinates_flat = [coord for sublist in geometry.coordinates().getInfo() for coord in sublist]
    # Convert the flattened coordinates to ee.Geometry.Polygon
    ee_polygon = ee.Geometry.Polygon(coordinates_flat)
    #########################################################
    # Filter the image collection based on the date range and location
    collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA') \
        .filterDate(start, end) \
        .filterBounds(ee_polygon)
    # Ensure there are some images in the collection
    # Check if there are any images in the collection
    if collection.size().getInfo() > 0:
        first_image = collection.first()  # Get the first image
    else:
        print(f"No images found for district: {districts} and date: {selected_date}")
        return None  # Or handle the case as needed
    all_band_names = first_image.bandNames().getInfo()
    # Select all bands from each image in the collection
    all_bands_collection = collection.select(all_band_names)
    # Clip the image collection to the polygon
    clipped_image = all_bands_collection.max().clip(ee_polygon)
    # Cast all bands to Float32
    clipped_image = clipped_image.toFloat()  # Ensure data type consistency
    # Set the export parameters for GeoTIFF
    download_url = clipped_image.getDownloadURL({
        'scale':  1113.2,
        'region': ee_polygon,
        'fileFormat': 'GeoTIFF'
    })


    return (download_url)
def plot_prediction(prediction,selected_date):
    methane = (prediction * (0.706 * 10000 * 1113.2 * 1113.2) / (1000 * 1000000))
    methane_image = methane.reshape(first_height, first_width)
    methane_image[methane_image == 0] = np.nan
    # Normalize the methane_image between 0 and 1
    normalized_methane_image = methane_image
    # Create a heatmap trace
    heatmap = go.Heatmap(
        z=normalized_methane_image,
        colorscale='Jet', # Choose your desired colorscale
        zmin=np.nanmin(methane)-np.nanmin(methane)/40,
        zmax=np.nanmax(methane)+np.nanmin(methane)/40,
    )
    # Create the figure layout
    layout = go.Layout(
        title=f'Methane concentration Kg on {selected_date}',
        xaxis=dict(title='Km', scaleanchor='y', scaleratio=1),
        yaxis=dict(title='km'),
        coloraxis_colorbar=dict(
            title=f'Methane concentration Kg on {selected_date}',
        )
    )
    # Create the figure
    fig = go. Figure(data=[heatmap], layout=layout)
    # Show the figure
    return fig
if page == "Methane for Selected Date":
    min_date = datetime(2014, 1, 1)
    max_date = datetime(2023,12,1)
    selected_start_date = st.date_input("Select start month and year:", min_value=min_date, max_value=max_date, value=min_date)
    # Extract the selected month and year from the selected date
    selected_start_month = selected_start_date.month
    selected_start_year = selected_start_date.year
    selected_start_time = f'1-{selected_start_month }-{selected_start_year}'
    selected_date=selected_start_time
    state=state_dropdown
    districts=district_dropdown
    download_url=data_landsat8_download(districts, selected_date,district_id,state )

    response = requests.get(download_url)
    zip_file = zipfile.ZipFile(BytesIO(response.content))
    file_names = zip_file.namelist()

    # Function to process GeoTIFF files
    first_height=None
    first_width=None
    def process_geotiffs():
        global first_height, first_width
        flattened_arrays = []
        for file_name in file_names:
            if not file_name.endswith('.tif'):
                continue
            with zip_file.open(file_name) as src:
                with rasterio.open(BytesIO(src.read())) as tif_dataset:
                    data = tif_dataset.read(1)
                    flattened_arrays.append(data.flatten())
                    if first_height is None and first_width is None:
                        first_height = tif_dataset.height
                        first_width = tif_dataset.width
        final_array = np.column_stack(flattened_arrays)
        return final_array,first_height,first_width


    final_array,first_height,first_width = process_geotiffs()
    print("Shape of the final flattened array:", final_array.shape)
    print("Initial height of the first GeoTIFF file:", first_height)
    print("Initial width of the first GeoTIFF file:", first_width)
    final_array = np.nan_to_num(final_array, nan=-9999)

    # Machine learning model loading and predictions
    model_path = 'data/model_full_all_band.joblib'
    model = joblib.load(model_path)
    prediction = model.predict(final_array)
    prediction[prediction == prediction[0]] = np.nan
    prediction = np.nan_to_num(prediction, nan=0)

    methane = (prediction * (0.706 * 10000 * 1113.2 * 1113.2) / (1000 * 1000000))
    total_methane = (prediction * (0.706 * 10000 * 1113.2 * 1113.2) / (1000 * 1000000)).sum()

    # Methane visualization
    methane_image = methane.reshape(first_height, first_width)
    methane_image[methane_image == 0] = np.nan


    st.plotly_chart(plot_prediction(prediction,selected_date))















############### monthly images
if page == "Monthly Methane for Selected Year":

    # Define the minimum and maximum dates for the date input widget
    min_date = datetime(2014, 1, 1)
    max_date = datetime(2023, 12, 1)


    years = list(range(2014, 2025))  # Create a list of years from 2014 to 2024
    selected_year = st.selectbox("Select a year:", years)

    # Loop over all months of the selected year and show the methane prediction for each month
    for month in range(1, 13):
        # Create the selected date string for the current month
        selected_date = f'1-{month}-{selected_year}'
       
        selected_date=selected_start_time
        state=state_dropdown
        districts=district_dropdown
        download_url=data_landsat8_download(districts, selected_date,district_id,state )

        response = requests.get(download_url)
        zip_file = zipfile.ZipFile(BytesIO(response.content))
        file_names = zip_file.namelist()

        # Function to process GeoTIFF files
        first_height=None
        first_width=None
        def process_geotiffs():
            global first_height, first_width
            flattened_arrays = []
            for file_name in file_names:
                if not file_name.endswith('.tif'):
                    continue
                with zip_file.open(file_name) as src:
                    with rasterio.open(BytesIO(src.read())) as tif_dataset:
                        data = tif_dataset.read(1)
                        flattened_arrays.append(data.flatten())
                        if first_height is None and first_width is None:
                            first_height = tif_dataset.height
                            first_width = tif_dataset.width
            final_array = np.column_stack(flattened_arrays)
            return final_array,first_height,first_width


        final_array,first_height,first_width = process_geotiffs()
        print("Shape of the final flattened array:", final_array.shape)
        print("Initial height of the first GeoTIFF file:", first_height)
        print("Initial width of the first GeoTIFF file:", first_width)
        final_array = np.nan_to_num(final_array, nan=-9999)

        # Machine learning model loading and predictions
        model_path = 'data/model_full_all_band.joblib'
        model = joblib.load(model_path)
        prediction = model.predict(final_array)
        prediction[prediction == prediction[0]] = np.nan
        prediction = np.nan_to_num(prediction, nan=0)

        methane = (prediction * (0.706 * 10000 * 1113.2 * 1113.2) / (1000 * 1000000))
        total_methane = (prediction * (0.706 * 10000 * 1113.2 * 1113.2) / (1000 * 1000000)).sum()

        # Methane visualization
        methane_image = methane.reshape(first_height, first_width)
        methane_image[methane_image == 0] = np.nan


        st.plotly_chart(plot_prediction(prediction,selected_date))



if page =="How to use":
    # Define the data
    data = {
        "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "QA_PIXEL", "QA_RADSAT", "SAA", "SZA", "VAA", "VZA"],
        "correlations": ["Variable 1", "Variable 2", "Variable 3", "Variable 4", "Variable 5", "Variable 6", "Variable 7", "Variable 8", "Variable 9", "Variable 10", "Variable 11", "Variable 12", "Variable 13", "Variable 14", "Variable 15", "Variable 16", "Variable 17"],
        "Pearson's correlation": [-0.218611918, -0.203752113, -0.133021568, -0.04427143, -0.022406529, 0.280999896, 0.307238305, -0.105190125, -0.181394252, 0.092229457, 0.092482866, -0.276202475, 0.005141523, -0.110569172, -0.363738064, -0.132183607, -0.140620997],
        "Spearman's rank correlation": [-0.070733236, -0.039015021, 0.048031542, 0.092475551, 0.073135039, 0.367350134, 0.403802357, 0.063818942, -0.300486897, 0.250990995, 0.265665726, -0.364998209, 0.009272817, -0.313637911, -0.514163062, -0.202873772, -0.172157977]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Display table using Streamlit
    st.table(df)
    # Define the data
    data1 = {
        "Name": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "QA_PIXEL", "QA_RADSAT", "SAA", "SZA", "VAA", "VZA"],
        "Pixel Size": ["30 meters", "30 meters", "30 meters", "30 meters", "30 meters", "30 meters", "30 meters", "15 meters", "30 meters", "30 meters", "30 meters", "30 meters", "30 meters", "30 meters", "30 meters", "30 meters", "30 meters"],
        "Wavelength": ["0.43 - 0.45 μm", "0.45 - 0.51 μm", "0.53 - 0.59 μm", "0.64 - 0.67 μm", "0.85 - 0.88 μm", "1.57 - 1.65 μm", "2.11 - 2.29 μm", "0.52 - 0.90 μm", "1.36 - 1.38 μm", "10.60 - 11.19 μm", "11.50 - 12.51 μm", "", "", "", "", "", ""],
        "Description": ["Coastal aerosol", "Blue", "Green", "Red", "Near infrared", "Shortwave infrared 1", "Shortwave infrared 2", "Band 8 Panchromatic", "Cirrus", "Thermal infrared 1, resampled from 100m to 30m", "Thermal infrared 2, resampled from 100m to 30m", "Landsat Collection 2 QA Bitmask", "Radiometric saturation QA", "Solar Azimuth Angle", "Solar Zenith Angle", "View Azimuth Angle", "View Zenith Angle"]
    }

    # Create DataFrame
    df1 = pd.DataFrame(data1)

    # Display table using Streamlit
    st.table(df1)


    # Header
    st.title("CH4 Column Mass Calculation")

    # Information
    st.markdown("""
    The formula used to convert `CH4_column_volume_mixing_ratio_dry_air` from ppb to kg/m^2 comes from basic principles of chemistry and atmospheric sciences. It combines multiple concepts and constants to arrive at the final expression. Here's a breakdown of the source of each part:

    1. **Molecule conversion:**
        - **Units:** CH4_column_volume_mixing_ratio_dry_air is given in ppb. We need to convert it to a number concentration (molecules per unit volume).
        - **Source:** This conversion uses the definition of ppb, which stands for **parts per billion**. One ppb means there is 1 molecule of CH4 for every billion molecules of dry air. This conversion factor is therefore 10^-9 (as used in the formula).

    2. **Column averaging:**
        - **Units:** The CH4 value represents an **average concentration** across the atmospheric column.
        - **Source:** We need to multiply by the number of molecules in the column to get the total amount of CH4. This value, denoted by `N_molecules_per_m2`, depends on the specific atmospheric profile and location. It's typically found in atmospheric science references or by querying the data itself.

    3. **Number concentration to mass:**
        - **Units:** We have the number of CH4 molecules per unit area. We need to convert this to mass.
        - **Sources:**
            - **Avogadro's constant:** This constant (6.022 x 10^23 molecules/mol) represents the number of molecules in one mole of any substance. It allows us to convert from molecules to moles.
            - **Molar mass of CH4:** This is the mass of one mole of CH4 (16 g/mol). It allows us to convert from moles to mass.

    4. **Height of the atmospheric column:**
        - **Units:** The average CH4 concentration is given for a certain column height.
        - **Source:** This value is typically around 10 km for the total troposphere. However, depending on the data specifics, you might need to use a different value based on the actual column considered.

    Therefore, the formula combines these components from various sources:

    - Units conversions and definitions (ppb, Avogadro's constant)
    - Atmospheric science concepts (column averaging, typical column height)
    - Chemical properties (molar mass)

    **mass_per_m2 = (CH4_column_volume_mixing_ratio_dry_air_ppb * 10^-9) * N_molecules_per_m2 * 6.022e23 * 16e-3 g/mol * 10e3 m.**

    **Example Calculation:**

    Assuming `CH4_column_volume_mixing_ratio_dry_air_ppb` is 1900 ppb, `N_molecules_per_m2` is 10^25 molecules/m^2, and the column height is 10 km:

    **mass_per_m2 = (1900 * 10^-9) * 10^25 * 6.022e23 * 16e-3 g/mol * 10e3 m**

    **= 0.00607 kg/m^2**
                
    **Mass in kg per pixel=0.00607*(pixel size in m)^2 **
    - ** here we are using pixel size 1113.2m X 1113.2m **
    -** so **
                **=0.0607*1113.2*1113.2**
                **=75,220.30 Kg**
    For converting parts per million (ppm) to parts per billion (ppb), you can use the [Lenntech PPM to PPB Converter](https://www.lenntech.com/calculators/ppm/converter-parts-per-million.htm).
    """)

    st.header("How to convert PPB to Kg")
    st.write('mass kg =(CH4 column volume mixing ratio dry air×16.04)×(pixel size in meter)^2')
























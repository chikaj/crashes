import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os
    import io

    functions_dir = os.path.abspath('./functions')
    if functions_dir not in sys.path:
        sys.path.append(functions_dir)

    import marimo as mo
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point
    from scipy.stats import skew, kurtosis
    import pingouin as pg
    import openlayers as ol
    import utils

    import folium
    import pandas as pd
    import geopandas as gpd
    from folium.plugins import MarkerCluster
    import matplotlib.pyplot as plt
    import seaborn as sns

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    return Point, gpd, mo, np, pd, plt, sns, utils


@app.cell
def _(mo):
    mo.vstack([
        mo.md('# Utah Crash Analysis'),
        mo.image('img/traffic.jpg')
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Project Abstract

    Crashes are bad. They cost a lot...in healthcare and lost lives, property damage and reduced productivity (due to delays). Everyone wants to reduce them, but to do so we need to know what the biggest contributors are to crashes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Data sources
    1. [Crash](./data/Utah_Crash_Data_Dummy.csv) data
    2. [AADT](./data/aadt.geojson) data
    3. [Crash](./data/Utah_Crash_Data_AADT.csv) combined data
    """)
    return


@app.cell
def _(pd):
    # dtype_map = {
    #     'CRASH_ID': 'Int32',
    #     'CRASH_SEVERITY_ID': 'Int32',
    #     'NEW_CITY': 'Int32',
    #     'NEW_COUNTY_NAME': 'Int32',
    #     'LONG_UTM_X': 'float64',
    #     'MAIN_ROAD_NAME': 'string',
    #     'CITY': 'string',
    #     'COUNTY_NAME': 'string'
    # }

    # date_columns_to_parse = [
    #     'CRASH_DATETIME'
    # ]

    # crashes_csv = pd.read_csv('data/Utah_Crash_Data_Dummy.csv', dtype=dtype_map, parse_dates=date_columns_to_parse, low_memory=False)
    # crashes_csv = pd.read_csv('data/Utah_Crash_Data_Dummy.csv')
    crashes_csv = pd.read_csv('data/Utah_Crash_Data_AADT.csv')
    return (crashes_csv,)


@app.cell
def _(crashes_csv, mo):
    # List of problematic column indices
    problem_col_names = [crashes_csv.columns[i] for i in [0, 16, 17, 19, 52]]

    mo.vstack([
        mo.md('## The following columns have **mixed types** issues'),
        problem_col_names
    ])
    return


@app.cell
def _(crashes_csv, mo):
    mo.vstack([
        mo.md('### Raw data (i.e., needs to be cleaned)'),
        crashes_csv
    ])
    return


@app.cell
def _(crashes_csv, mo, utils):
    mo.vstack([
        mo.md('### Raw Data: Summary Statistics'),
        utils.full_extended_describe(crashes_csv)
    ])
    return


@app.cell
def _(crashes_csv, mo):
    df_widget0 = mo.ui.dataframe(crashes_csv)

    mo.vstack([
        mo.md('''
            ### Cleaning: Remove nulls from CRASH_DATETIME
            '''),
        df_widget0
    ])
    return (df_widget0,)


@app.cell
def _(mo):
    mo.vstack([
        mo.md('### Cleaning: data type conversion'),
        mo.md('''
        **CRASH_ID**: 'Int32', **CRASH_DATETIME**: np.datetime64, **YEAR**: "Int32", **MILEPOINT**: np.float64, **LAT_UTM_Y**: np.float64, **LONG_UTM_X**: np.float64, **CRASH_SEVERITY_ID**: 'Int32', **Serious**: "Int32", **WORK_ZONE_RELATED**: "Int32"
        ''')
    ])
    return


@app.cell
def _(df_widget0, np, utils):
    # 1. Define the complete map of ALL desired final types
    target_conversion_map = {
        'CRASH_ID': 'Int32',
        'CRASH_DATETIME': np.datetime64, # datetime works here
        'YEAR': "Int32",
        'MONTH': "Int32",
        'DAY': "Int32",
        'HOUR': "Int32",
        'MINUTES': "Int32",
        'DAYTIME': "Int32",
        'DOW': "Int32",
        'Monday': "Int32",
        'Tuesday': "Int32",
        'Wednesday': "Int32",
        'Thursday': "Int32",
        'Friday': "Int32",
        'Saturday': "Int32",
        'Sunday': "Int32",
        'ROUTE': "Int32",
        'MILEPOINT': np.float64,
        'LAT_UTM_Y': np.float64,
        'LONG_UTM_X': np.float64,
        'MAIN_ROAD_NAME': 'string',
        'CITY': 'string',
        'COUNTY_NAME': 'string',
        'CRASH_SEVERITY_ID': 'Int32',
        'Serious': "Int32",
        'WORK_ZONE_RELATED': "Int32",
        'PEDESTRIAN_INVOLVED': "Int32",
        'BICYCLIST_INVOLVED': "Int32",
        'MOTORCYCLE_INVOLVED': "Int32",
        'IMPROPER_RESTRAINT': "Int32",
        'UNRESTRAINED': "Int32",
        'DUI': "Int32",
        'INTERSECTION_RELATED': "Int32",
        'NEW_CITY': 'Int32',
        'NEW_COUNTY_NAME': 'Int32',
    }

    # 2. Apply all transforms in one function call
    type_converted = utils.apply_type_conversion_map(
        df=df_widget0.value, # Start with the current widget value
        conversion_map=target_conversion_map,
        clean_and_convert_func=utils.clean_and_convert_column 
    )
    return (type_converted,)


@app.cell
def _(mo, type_converted, utils):
    mo.vstack([
        mo.md('### Cleaned Data: Summary Statistics'),
        utils.full_extended_describe(type_converted)
    ])
    return


@app.cell
def _(mo, type_converted):
    df_widget1 = mo.ui.dataframe(type_converted)

    mo.vstack([
        mo.md('### Cleaning: Remove nulls from lon/lat pairs, WORK_ZONE_RELATED and CRASH_SEVERITY_ID'),
        df_widget1
    ])
    return (df_widget1,)


@app.cell
def _(df_widget1, mo, utils):
    mo.vstack([
        mo.md('### Cleaned Data: Summary Statistics'),
        utils.full_extended_describe(df_widget1.value)
    ])
    return


@app.cell
def _(df_widget1, mo):
    filtered = df_widget1.value

    mo.vstack([
        mo.md('### Cleaned data'),
        filtered
    ])
    return (filtered,)


@app.cell
def _(Point, filtered, gpd):
    # Create a GeoSeries of Point objects
    # We create a list of Point objects from the coordinates
    geometry = [
        Point(xy) 
        for xy in zip(filtered['LONG_UTM_X'], filtered['LAT_UTM_Y'])
    ]

    # Create the GeoDataFrame
    crash_layer_utm = gpd.GeoDataFrame(
        filtered, 
        geometry=geometry # Assign the list of Point objects as the geometry column
    )

    # Assign the CRS (UTM Zone 12N)
    crash_layer_utm.set_crs(epsg=32612, inplace=True)

    # You can optionally convert it to a standard geographic CRS (WGS 84, Lat/Lon) 
    # for plotting in web maps or general use:
    crash_layer_wgs = crash_layer_utm.to_crs(epsg=4326) # EPSG: 4326 is standard Lat/Lon
    return (crash_layer_wgs,)


@app.cell
def _(gpd):
    aadt = gpd.read_file('./data/AADT_Unrounded.shp')
    aadt
    return (aadt,)


@app.cell
def _(aadt):
    aadt.crs
    return


@app.cell
def _(aadt):
    aadt.to_crs(epsg=4326)
    return


@app.cell
def _(aadt):
    aadt.columns
    return


@app.cell
def _(aadt):
    aadt_columns = [
        'OBJECTID',
        'Station', 
        'AADT2020',
        'AADT2019',
        'AADT2018',
        'AADT2017',
        'AADT2016',
        'geometry'
    ]

    aadt_subset = aadt[aadt_columns]
    aadt_subset
    return


@app.cell
def _(aadt, crash_layer_wgs, plt):
    # --- STEP 0: Ensure CRS Match (Crucial!) ---
    # If your point_layer is in EPSG:4326 (Lat/Lon) and your aadt is in UTM (e.g., EPSG:32612), 
    # you must re-project one to match the other. Let's project AADT to match the points.
    aadt_wgs = aadt.to_crs(epsg=4326)

    # --- STEP 1: Set up the Plot Canvas (Matplotlib Figure and Axes) ---
    # Create a figure (the entire canvas) and a set of axes (the map area)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # --- STEP 2: Plot the Base Layer (Roads) ---
    # Plot the AADT road segments first. Store the result in 'ax'.
    aadt_wgs.plot(
        ax=ax,
        linewidth=1.5,
        edgecolor='gray',
        label='Road Segments'
    )

    # --- STEP 3: Plot the Second Layer (Points) ---
    # Plot the crash points ON THE SAME AXES (ax=ax)
    crash_layer_wgs.plot(
        ax=ax,
        marker='o',
        color='red',
        markersize=10,
        alpha=0.6,
        label='Crash Locations'
    )

    # --- STEP 4: Add Map Elements and Finalize ---
    ax.set_title("Utah Crashes on AADT Road Network", fontsize=16)
    ax.set_xlabel("Longitude" if crash_layer_wgs.crs.to_epsg() == 4326 else "Easting (m)")
    ax.set_ylabel("Latitude" if crash_layer_wgs.crs.to_epsg() == 4326 else "Northing (m)")

    # Add a simple legend to distinguish the layers
    ax.legend(loc='lower left')

    # Display the map
    plt.show()
    return


@app.cell
def _(crash_layer_wgs):
    crash_layer_wgs.columns
    return


@app.cell
def _(crash_layer_wgs, gpd, np, pd, plt):
    def generate_all_histograms(df: gpd.GeoDataFrame, max_unique_bins=50, max_plots_per_figure=16):
    
        # ... (Steps 1 & 2: Column filtering remains the same) ...
        ignore_columns = [
            'CRASH_ID', 'MAIN_ROAD_NAME', 'CITY', 'COUNTY_NAME', 
            'geometry', 'CRASH_DATETIME', 'LAT_UTM_Y', 'LONG_UTM_X'  
        ]
    
        plottable_columns = []
        for col in df.columns:
            if col in ignore_columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                plottable_columns.append(col)
            elif pd.api.types.is_integer_dtype(df[col].dtype) and df[col].nunique() <= max_unique_bins:
                plottable_columns.append(col)
            elif pd.api.types.is_object_dtype(df[col].dtype) and df[col].nunique() <= 10:
                 plottable_columns.append(col)

        print(f"Found {len(plottable_columns)} columns suitable for histogram plotting.")
    
        # --- 3. Batch Plotting (With Fix) ---
        num_cols = len(plottable_columns)
    
        for i in range(0, num_cols, max_plots_per_figure):
        
            batch_cols = plottable_columns[i:i + max_plots_per_figure]
            num_in_batch = len(batch_cols)
        
            rows = int(np.ceil(np.sqrt(num_in_batch)))
            cols = int(np.ceil(num_in_batch / rows))
        
            fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
            axes = axes.flatten() 
        
            for j, col in enumerate(batch_cols):
                ax = axes[j]
            
                # --- FIX: Check if the Series is empty before plotting ---
            
                # Check for categorical/binary columns
                if df[col].nunique() <= 10 or col in ['YEAR', 'MONTH', 'DAY', 'HOUR', 'DOW']:
                
                    plot_data = df[col].value_counts().sort_index()
                
                    if not plot_data.empty:
                        # Bar plot for categories/flags
                        plot_data.plot(kind='bar', ax=ax, color='skyblue')
                        ax.tick_params(axis='x', rotation=0) 
                    else:
                        ax.text(0.5, 0.5, "No data for bar plot", 
                                transform=ax.transAxes, ha='center', va='center')
                    
                # Check for continuous data columns
                else:
                    # Use .dropna() to prevent plotting NaN values, which can also cause issues
                    plot_data = df[col].dropna() 
                
                    if len(plot_data) > 0:
                        # Standard histogram for continuous data
                        plot_data.hist(ax=ax, bins=30, color='teal', edgecolor='black')
                    else:
                        ax.text(0.5, 0.5, "No data for histogram", 
                                transform=ax.transAxes, ha='center', va='center')
            
                ax.set_title(col, fontsize=10)
                ax.set_xlabel('') 
            
            # Hide any unused subplots
            for k in range(num_in_batch, rows * cols):
                fig.delaxes(axes[k])
            
            plt.tight_layout()
            plt.show()

    # --- Execution ---
    # Now, the function will safely handle any column that results in zero data points 
    # and will print a message instead of crashing.

    # --- Execution ---
    # You need to ensure crash_layer_wgs is loaded in a preceding Marimo cell.
    # The function will run only if the GeoDataFrame has content.

    if 'crash_layer_wgs' in locals() and len(crash_layer_wgs) > 0:
        generate_all_histograms(crash_layer_wgs)
    else:
        print("GeoDataFrame 'crash_layer_wgs' not found or is empty. Cannot generate histograms.")
    return


@app.cell
def _(crash_layer_wgs, gpd, np, pd, plt, sns):
    # Assuming your GeoDataFrame is named 'crash_layer_wgs'

    def generate_bivariate_box_plots(df: gpd.GeoDataFrame, category_columns: list, max_plots_per_figure=9):
        """
        Generates box plots for all suitable numeric columns against specified 
        categorical columns.

        Args:
            df (gpd.GeoDataFrame): The input GeoDataFrame (e.g., crash_layer_wgs).
            category_columns (list): Columns used as the x-axis categories (e.g., ['DOW', 'CRASH_SEVERITY_ID']).
            max_plots_per_figure (int): Maximum number of box plots per Matplotlib figure.
        """
    
        # --- 1. Identify Target Numeric Columns (Y-Axis) ---
        # We select numeric columns suitable for distributional analysis.
        numeric_ignore = [
            'CRASH_ID', 'LAT_UTM_Y', 'LONG_UTM_X', 'geometry',
            # Exclude binary flags that don't make sense as a Y-axis (e.g., Tuesday, DUI)
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 
            'Serious', 'WORK_ZONE_RELATED', 'PEDESTRIAN_INVOLVED', 'BICYCLIST_INVOLVED',
            'MOTORCYCLE_INVOLVED', 'IMPROPER_RESTRAINT', 'UNRESTRAINED', 'DUI', 
            'INTERSECTION_RELATED', 'WILD_ANIMAL_RELATED', 'DOMESTIC_ANIMAL_RELATED', 
            'OVERTURN_ROLLOVER', 'COMMERCIAL_MOTOR_VEH_INVOLVED', 'TEENAGE_DRIVER_INVOLVED',
            'OLDER_DRIVER_INVOLVED', 'NIGHT_DARK_CONDITION', 'SINGLE_VEHICLE',
            'DISTRACTED_DRIVING', 'DROWSY_DRIVING', 'ROADWAY_DEPARTURE'
        ]
    
        # Identify true numeric columns for the y-axis
        numeric_cols = [
            col for col in df.columns 
            if pd.api.types.is_numeric_dtype(df[col].dtype) and col not in numeric_ignore
        ]
    
        # --- 2. Create Combinations and Plot ---
    
        # Create combinations of (Categorical_X, Numeric_Y)
        plot_combinations = [
            (cat, num) 
            for cat in category_columns 
            for num in numeric_cols
            # Only plot if the category column has data and is not too wide
            if df[cat].nunique() > 1 and df[cat].nunique() <= 20
        ]

        print(f"Found {len(numeric_cols)} numeric columns and generating {len(plot_combinations)} bivariate box plots...")
    
        for i in range(0, len(plot_combinations), max_plots_per_figure):
        
            batch = plot_combinations[i:i + max_plots_per_figure]
            num_in_batch = len(batch)
        
            # Calculate optimal grid size (e.g., 3x3 for 9 plots)
            rows = int(np.ceil(np.sqrt(num_in_batch)))
            cols = int(np.ceil(num_in_batch / rows))

            fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
            axes = axes.flatten()
        
            for j, (cat_col, num_col) in enumerate(batch):
                ax = axes[j]
            
                # --- Check for empty data before plotting ---
                # Drop rows where the numeric column is NaN, as boxplot requires non-null data
                plot_df = df.dropna(subset=[num_col, cat_col])
            
                if len(plot_df) > 0:
                    # --- Seaborn Box Plot --- 
                    sns.boxplot(
                        x=cat_col, 
                        y=num_col, 
                        data=plot_df, 
                        ax=ax,
                        palette='viridis', 
                        showfliers=False # Optionally hide extreme outliers for clearer view
                    )
                
                    ax.set_title(f'{num_col} by {cat_col}', fontsize=10)
                    ax.set_xlabel(cat_col, fontsize=8)
                    ax.set_ylabel(num_col, fontsize=8)
                
                    # Rotate x-labels if categories are too long
                    if plot_df[cat_col].nunique() > 5:
                         ax.tick_params(axis='x', rotation=45)
                else:
                    ax.text(0.5, 0.5, f"No data to plot for {num_col} vs {cat_col}", 
                            transform=ax.transAxes, ha='center', va='center')


            # Hide any unused subplots
            for k in range(num_in_batch, rows * cols):
                fig.delaxes(axes[k])
            
            plt.tight_layout()
            plt.show()

    # --- Example Execution ---

    # 1. Define the categorical columns you want to use as the x-axis
    # These columns must be low-cardinality flags or codes.
    categorical_target_cols = [
        'CRASH_SEVERITY_ID', 
        'DOW', 
        'DAYTIME'
    ]

    # Ensure your GeoDataFrame is loaded:
    if 'crash_layer_wgs' in locals() and len(crash_layer_wgs) > 0:
        generate_bivariate_box_plots(crash_layer_wgs, categorical_target_cols)
    else:
        print("GeoDataFrame 'crash_layer_wgs' not loaded or is empty.")

    print("Bivariate box plot generation script defined. Ready for execution on 'crash_layer_wgs'.")
    return


@app.cell
def _(crash_layer_wgs, gpd, plt, sns):
    # Assuming your GeoDataFrame is named 'crash_layer_wgs'

    def plot_severity_by_dow(df: gpd.GeoDataFrame):
        """
        Generates a bar chart showing the count of each CRASH_SEVERITY_ID 
        grouped by Day of Week (DOW).

        Args:
            df (gpd.GeoDataFrame): The input crash data.
        """
    
        # 1. Define the order for the days of the week (DOW is often a string/object)
        # This ensures the days are plotted correctly (Mon, Tue, Wed, etc.) instead of alphabetically.
        day_order = [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 
            'Friday', 'Saturday', 'Sunday'
        ]

        # 2. Define the order for severity IDs (Optional, but makes the chart clearer)
        # Assuming lower number is less severe.
        severity_order = sorted(df['CRASH_SEVERITY_ID'].unique())
    
        # Check if essential columns exist and data is present
        if not all(col in df.columns for col in ['DOW', 'CRASH_SEVERITY_ID']) or len(df) == 0:
            print("Required columns ('DOW', 'CRASH_SEVERITY_ID') are missing or DataFrame is empty. Skipping plot.")
            return

        # 3. Create the figure and axes
        plt.figure(figsize=(10, 6))

        # 4. Generate the Plot (Countplot)
        # Use 'DOW' for the x-axis, count the frequency (implicitly on y), 
        # and use 'CRASH_SEVERITY_ID' for the hue (color/stacking).
        sns.countplot(
            x='DOW', 
            hue='CRASH_SEVERITY_ID', 
            data=df, 
            order=day_order,
            hue_order=severity_order,
            palette='viridis' # Use a color palette to differentiate severities
        )
    
        # 5. Add labels and title
        plt.title('Crash Count by Day of Week and Severity', fontsize=14)
        plt.xlabel('Day of Week', fontsize=12)
        plt.ylabel('Count of Crashes', fontsize=12)
        plt.xticks(rotation=45, ha='right')
    
        # Improve legend title
        plt.legend(title='Severity ID', loc='upper right')
    
        plt.tight_layout()
        plt.show()

    # --- Example Execution ---
    # You need to ensure crash_layer_wgs is loaded in a preceding Marimo cell.

    if 'crash_layer_wgs' in locals() and len(crash_layer_wgs) > 0:
        plot_severity_by_dow(crash_layer_wgs)
    else:
        print("GeoDataFrame 'crash_layer_wgs' not loaded or is empty.")

    print("Bar chart script defined. Ready for execution on 'crash_layer_wgs'.")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

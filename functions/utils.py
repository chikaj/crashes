from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd


def extended_describe(df):
    """
    Computes descriptive statistics, including missing values, skewness, and kurtosis
    for numerical columns in a DataFrame.
    """
    # 1. Select numerical columns for analysis
    df_num = df.select_dtypes(include=np.number)

    if df_num.empty:
        return "No numerical columns found to compute extended statistics."

    # 2. Base descriptive statistics
    d = df_num.describe()

    # 3. Calculate additional statistics
    # Missing/Nulls
    total_rows = len(df)
    missing = df_num.isnull().sum()
    missing.name = "missing"
    # Calculate missing percentage
    missing_pct = ((missing / total_rows) * 100).round(4)
    missing_pct.name = "missing_pct"

    # Skewness
    skewness = df_num.skew(numeric_only=True)
    skewness.name = "skew"

    # Kurtosis
    kurt = df_num.kurt(numeric_only=True)
    kurt.name = "kurtosis"

    # 4. Combine all statistics
    # Concatenate the new series (converted to single-row DataFrames) to the base describe DataFrame
    stats_df = pd.concat(
        [
            d,
            missing.to_frame().T,
            missing_pct.to_frame().T,
            skewness.to_frame().T,
            kurt.to_frame().T,
        ]
    )

    # Reorder the index for better presentation
    new_order = [
        "count",
        "missing",
        "missing_pct",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "skew",
        "kurtosis",
    ]

    stats_df = stats_df.reindex(new_order)

    return stats_df.T


def full_extended_describe(df):
    """
    Computes a full set of descriptive statistics for all columns (numerical and categorical),
    including missing values, skewness, and kurtosis.

    - For numerical columns: Calculates mean, std, min, max, quartiles, skewness, and kurtosis.
    - For categorical/object columns: Calculates unique, top, and freq.
    - For all columns: Calculates count, missing count, and missing percentage.

    The result is transposed so features are rows and statistics are columns.
    """

    # --- 1. Base descriptive statistics (includes all dtypes) ---
    # We use include='all' to get statistics for all data types.
    # Transposing immediately makes the statistics the columns and features the index.
    d = df.describe(include="all").T

    # --- 2. Calculate Missing/Nulls ---
    total_rows = len(df)
    missing_series = df.isnull().sum().rename("missing")
    missing_pct_series = (
        ((missing_series / total_rows) * 100).round(4).rename("missing_pct")
    )

    # --- 3. Create a new DataFrame with missing stats ---
    # missing_stats = pd.DataFrame({"missing": missing, "missing_pct": missing_pct})

    # --- 3. Calculate Dtype (The new addition) ---
    # Convert dtypes (e.g., int64, object) to a simple string Series
    dtype_series = df.dtypes.astype(str).rename("dtype")

    # --- 3. Calculate Skewness and Kurtosis (only for numerical columns) ---
    # .skew() and .kurt() naturally ignore non-numerical data.
    skewness = df.skew(numeric_only=True)
    kurt = df.kurt(numeric_only=True)

    # Create Series that includes all original column names and fill in numerical stats.
    # Non-numerical columns will default to NaN, which is the desired behavior.
    skew_series = pd.Series(index=df.columns, name="skew", dtype=float)
    skew_series.update(skewness)

    kurt_series = pd.Series(index=df.columns, name="kurtosis", dtype=float)
    kurt_series.update(kurt)

    # --- 4. Combine all statistics ---
    # Concatenate the new stats with the base describe, joining on the column index (feature names).
    stats_df = pd.concat(
        [d, dtype_series, missing_series, missing_pct_series, skew_series, kurt_series],
        axis=1,
    )

    # --- 5. Define final column order ---
    final_cols = [
        "dtype",
        "count",
        "missing",
        "missing_pct",
        "unique",
        "top",
        "freq",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "skew",
        "kurtosis",
    ]

    # Reindex to ensure consistent order and filter only for columns that exist
    stats_df = stats_df.reindex(
        columns=[col for col in final_cols if col in stats_df.columns]
    )

    return stats_df


def find_unusual_type_values(df):
    """
    Finds values in each column whose data type is different from the most common
    data type in that column, excluding NaN values from the type count.

    Returns a dictionary where keys are column names and values are pandas Series
    containing the "unusual" values.
    """
    unusual_values_by_column = {}

    for col in df.columns:
        # Get the types of all NON-NULL values
        types = df[col].dropna().apply(type)

        # Skip if all values are null
        if types.empty:
            continue

        # Determine the most frequent (usual) type
        # .value_counts().index[0] returns the type with the highest count.
        usual_type = types.value_counts().index[0]

        # Create a boolean mask for values that are NOT of the usual type.
        unusual_mask = df[col].apply(type) != usual_type

        # Select and store the unusual values (dropping NaNs from the final output)
        unusual_values = df[col][unusual_mask].dropna()

        if not unusual_values.empty:
            unusual_values_by_column[col] = unusual_values

    return unusual_values_by_column


def clean_and_convert_column(
    df: pd.DataFrame, column_name: str, target_dtype: Union[str, type, np.dtype]
) -> pd.DataFrame:
    # --- 1. Determine the Final Target Data Type and Name ---
    dtype_name = str(target_dtype).split(".")[-1].lower()

    is_numeric = any(d in dtype_name for d in ["int", "float"])
    is_integer = any(d in dtype_name for d in ["int"])
    is_datetime = any(d in dtype_name for d in ["date", "time", "datetime"])

    # Map numpy/string types to pandas nullable types
    if is_integer and dtype_name in ["int32", "int64", "int"]:
        # Use pandas nullable integer (Int32, Int64)
        final_dtype = "Int64" if "64" in dtype_name else "Int32"
    elif is_datetime:
        final_dtype = "datetime64[ns]"
    elif dtype_name in ["str", "string", "object"]:
        final_dtype = "string"
    else:
        final_dtype = target_dtype

    # --- 2. String Cleaning ---
    if is_numeric:
        # Clean for numeric conversion
        df[column_name] = (
            df[column_name].astype(str).str.replace(r'[,"\'\$]', "", regex=True)
        )
    elif not is_datetime:
        # Ensure non-numeric/non-date targets are strings for safety
        df[column_name] = df[column_name].astype(str)

    # --- 3. Conversion & Floor (The Fixed Logic) ---

    # A. Handle Numeric Conversion (Always returns float64 or Series)
    if is_numeric:
        # Convert to numeric, coercing bad data to NaN. The column is now float64.
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

        # B. Apply Floor for Integer Targets (Fixes the safety casting error)
        if is_integer:
            # We must explicitly handle the conversion from float64 to int
            # by removing decimals before the final nullable cast.
            # We use a temporary assignment to prevent immediate errors,
            # and .floor() handles NaNs gracefully in modern pandas.
            df[column_name] = df[column_name].apply(np.floor)

    # C. Handle Datetime Conversion
    elif is_datetime:
        # Convert to datetime, coercing unparseable strings to NaT
        df[column_name] = pd.to_datetime(df[column_name], errors="coerce")

    # --- 4. Final Type Conversion ---
    # This applies the final nullable or specialized type.
    df[column_name] = df[column_name].astype(final_dtype)

    return df


def apply_type_conversion_map(
    df: pd.DataFrame,
    conversion_map: Dict[str, Union[str, type, np.dtype]],
    clean_and_convert_func: callable,
) -> pd.DataFrame:
    """
    Applies a dictionary of type conversions to multiple columns in a DataFrame
    in a single call, handling string cleaning, NaN coercion, and nullable types.

    Args:
        df (pd.DataFrame): The input DataFrame.
        conversion_map (Dict[str, Union[str, type, np.dtype]]): A dictionary
            where keys are column names and values are the target dtypes (e.g.,
            {'CRASH_ID': 'Int32', 'LONGITUDE': np.float64}).
        clean_and_convert_func (callable): The single-column cleaning function
            (e.g., clean_and_convert_column) to apply to each column.

    Returns:
        pd.DataFrame: The DataFrame with all specified columns converted.
    """

    # Create a copy to prevent the SettingWithCopyWarning in subsequent cleaning steps
    df_cleaned = df.copy()

    print(f"Applying conversions to {len(conversion_map)} columns...")

    for column_name, target_dtype in conversion_map.items():
        if column_name in df_cleaned.columns:
            # Call the single-column function for each column
            df_cleaned = clean_and_convert_func(df_cleaned, column_name, target_dtype)
        else:
            print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping.")

    print("Conversion batch complete.")
    return df_cleaned


def standardize_delimiter(
    input_filepath: str, temp_delimiter: str = "\t", final_delimiter: str = ","
) -> Tuple[str, int]:
    """
    Reads a file, replaces an undesirable delimiter (e.g., tabs) with the
    desired delimiter (e.g., commas), and returns the sanitized content
    as a single string. Reports the number of lines found.
    """
    sanitized_data = []
    line_count = 0

    with open(input_filepath, "r") as f:
        for line in f:
            line_count += 1
            # Replace all instances of the temporary delimiter with the final one
            sanitized_data.append(line.replace(temp_delimiter, final_delimiter))

    # Join the lines into a single string stream for pd.read_csv to read directly
    return "".join(sanitized_data), line_count

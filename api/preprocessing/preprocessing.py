import pandas as pd
import numpy as np


def roll_win(df : pd.DataFrame):
    df = df.reset_index()

    # Adding cyclic windows. Using sin and cos instead of month numbers e.g 1 for jan, 2 for feb, so the model will
    # see the jan and dec arent far apart at all like it will portray mathematically (1 and 12 are far)
    df["month"] = df["timestamp"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.drop(columns=["month"])
    df = df.set_index("timestamp")
    df = df.sort_index()
    for col in ['P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'P-JUS-CKGL', 'QGL']:
        # 10-minute statistics
        df[f"{col}_mean_10min"] = (
            df[col]
            .rolling(window="10min", min_periods=10)
            .mean()
        )
        df[f"{col}_var_10min"] = (
            df[col]
            .rolling(window="10min", min_periods=10)
            .var()
        )
        df[f"{col}_std_10min"] = (
            df[col]
            .rolling(window="10min", min_periods=10)
            .std()
        )
        df[f"{col}_range_10min"] = (
            df[col]
            .rolling(window="10min", min_periods=10)
            .max()
            -
            df[col]
            .rolling(window="10min", min_periods=10)
            .min()
        )
        # 120-minute statistics
        df[f"{col}_mean_120min"] = (
            df[col]
            .rolling(window="120min", min_periods=120)
            .mean()
        )
        df[f"{col}_std_120min"] = (
            df[col]
            .rolling(window="120min", min_periods=120)
            .std()
        )
    for col in ['P-TPT', 'P-MON-CKP', 'P-JUS-CKGL', 'QGL']:
        df[f"{col}_ewm"] = df[col].ewm(span=10).mean()  # EWM for pressure columns

        
    return df



def preproccesing(df : pd.DataFrame):
    """
    Creates additional columns to explain more information on the data.
    Arugments : dataframe
    
    Returns : processed dataframe
    """

    df = df.sort_index()
    
    df["T_P_divergence"] = df["T-TPT"] / (df["P-TPT"] + 1)
    

    df['choke_pressure_diff'] = df['P-MON-CKP'] - df['P-TPT']


    df['productivity_index'] = df['P-TPT'] / (df['QGL'] + 1)

    df['hydrate_risk'] = df['T-JUS-CKP'] - 20  # distance from hydrate threshold
    
    df['P_TPT_diff'] = df['P-TPT'].diff()
    
    # Find where current diff and previous diff have different signs (multiply to negative)
    is_sign_change = (df['P_TPT_diff'] * df['P_TPT_diff'].shift(1)) < 0
    
    df['P_TPT_sign_changes'] = is_sign_change.rolling(10, min_periods=1).sum()
    
    # 3. Vectorized Rolling Linear Regression Slope (Replaces np.polyfit)
    x_index = pd.Series(np.arange(len(df)), index=df.index)
    
    rolling_y = df['P-MON-CKP'].rolling(60, min_periods=10)
    
    rolling_x = x_index.rolling(60, min_periods=10)

    # Slope = Covariance(x, y) / Variance(x)
    df['P_MON_CKP_slope_60'] = (rolling_y.cov(x_index) / rolling_x.var())


    # 60  minute rolling window.
    df['P_TPT_std_60'] = df['P-TPT'].rolling('60min', min_periods=1).std()
    df['T_TPT_std_60'] = df['T-TPT'].rolling('60min', min_periods=1).std()
    
    # Creating stable Temp and Pressure column
    df['all_stable'] = (
        (df['P-TPT'].between(80, 140)) &
        (df['T-TPT'].between(50, 130)) &
        (df['P-MON-CKP'].between(10, 50))
    ).astype(int)
    
    # Stability streak finding the stabe
    df['stability_streak'] = df['all_stable'].groupby( 
        (df['all_stable'] != df['all_stable'].shift()).cumsum()
    ).cumcount()
    
    df = df.replace([np.inf, -np.inf], np.nan)

    return df



def process_file(file : str):
    """
    Data cleaning, resampling and feature engineering block
    Arguemnts: 
    file : file path. must be a csv file. File also contains oil dataset. 
    """
    
    df = pd.read_csv(f'{file}')
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # Convert the timestamp column to datetime format.
    df = df.sort_values("timestamp")
    df =  df.set_index("timestamp")

    # Strip extra spaces from column names immediately
    df.columns = df.columns.str.strip()

    # Remove T-JUS-CKGL entirely (it has zero values across all files)
    df = df.drop(columns=["P-PDG", "Instance_id"], errors='ignore')

    # ---------------------------------------------------------
    # THE DATA CLEANING BLOCK
    # ---------------------------------------------------------
    

    # 2. Interpolate Surface Sensors (10-minute limit = 600 seconds)
    cols_to_interpolate = ['P-MON-CKP', 'T-JUS-CKP', 'T-TPT', 'P-TPT']
    df[cols_to_interpolate] = df[cols_to_interpolate].interpolate(method='time', limit=600)
    
    # 5. Inject your Gas Lift logic here
    df["gas_lift_active"] = df["QGL"].notna().astype(int) 
    df["QGL"] = df["QGL"].fillna(0) 
    df["P-JUS-CKGL"] = df["P-JUS-CKGL"].fillna(0)

    # 6. Drop all remaining unrecoverable NaN rows
    df = df.dropna(subset=[
        'P-MON-CKP',
        'T-JUS-CKP',
        'T-TPT',
        'P-TPT'
    ])

    # ---------------------------------------------------------
    # THE CORRECTED RESAMPLING BLOCK
    # ---------------------------------------------------------
    # Resample data to 30sec groups
    feature = (
            df.drop(columns="class")
            .resample("30s")
            .agg('mean')
    )
    
    label = df["class"].resample("30s").last()
    df = feature.join(label)

    df = df.dropna(subset=["class"])

    return df

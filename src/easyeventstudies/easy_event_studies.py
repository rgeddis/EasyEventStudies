import requests
import zipfile
import io
import pandas as pd
from datetime import datetime
from typing import Tuple
import numpy as np
import statsmodels.regression.linear_model
import statsmodels.api as sm
from scipy.stats import t
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import pdb
from IPython.core.debugger import set_trace
import matplotlib.dates as mdates


def get_fama_french_daily_factors():
    """Downloads the Fama French daily factors from the Kenneth French data library and returns a pandas DataFrame."""
    url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # raises an HTTPError if the HTTP request returned an unsuccessful status code
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open('F-F_Research_Data_Factors_daily.CSV') as f:
                df = pd.read_csv(f, skiprows=4, skipfooter=2, engine='python')  # Skip header and footer
        return df

    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}. Please check your internet connection.")
        raise
    except zipfile.BadZipFile:
        print("Failed to unzip the data file. The file might be corrupted. Please contact Nicolas Roever at nicolas.roever@wiso.uni-koeln.de.")
        raise
    except pd.errors.ParserError:
        print("Failed to parse the CSV file. There might be an issue with the file format. Please contact Nicolas Roever at nicolas.roever@wiso.uni-koeln.de.")
        raise
    except Exception as e:
        print(f"A fatal error occurred. Please contact Nicolas Roever at nicolas.roever@wiso.uni-koeln.de. Error: {e}")
        raise


def clean_fama_french_factor_data(raw_data):
    """This function cleans the Fama French factor data set obtained from the Kenneth French data library.
    All numbers are in net returns."""

    cleaned_data = pd.DataFrame()
    cleaned_data['Date'] = raw_data['Unnamed: 0'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
    cleaned_data['Date'] = cleaned_data['Date'].dt.tz_localize(None)
    cleaned_data['Mkt-RF'] = raw_data['Mkt-RF'].astype(float) / 100
    cleaned_data['RF'] = raw_data['RF'].astype(float) / 100
    cleaned_data['SMB'] = raw_data['SMB'].astype(float) / 100
    cleaned_data['HML'] = raw_data['HML'].astype(float) / 100
    cleaned_data["Mkt"] = cleaned_data["Mkt-RF"] + cleaned_data["RF"]

    return cleaned_data

def clean_stock_returns(df):
    """
    Cleans stock returns data from yahoo finance.

    This function takes a DataFrame returned by yfinance.download and performs the following operations:
    1. Restructures the multi-index columns to have a separate Ticker column
    2. Ensures the 'Date' column is a datetime object
    3. Calculates the daily return as the percentage change from the 'Adj Close' column

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the stock returns data with multi-index columns
                          ['Price', 'Ticker']

    Returns:
    df (pandas.DataFrame): The cleaned DataFrame with single-level columns plus Ticker column
    """
    # Store the ticker name
    ticker = df.columns.get_level_values('Ticker')[0]
    
    # Get the first level column names and remove the 'Price' level entirely
    df.columns = [col[0] for col in df.columns]
    
    # Add Ticker column
    df['Ticker'] = ticker
    
    # Ensure the 'Date' column is datetime
    df.index = pd.to_datetime(df.index)
    df['Date'] = df.index
    df['Date'] = df['Date'].dt.tz_localize(None)
    
    # Reset the index
    df = df.reset_index(drop=True)
    
    # Calculate the daily return
    df['Daily_Return'] = df['Adj Close'].pct_change() 
    
    return df


def create_X_matrix(fama_french_data: pd.DataFrame, estimation_window: Tuple[int, int], event_date: datetime, mode: str = "market_model") -> np.ndarray:
    """
    Creates an X matrix based on the specified estimation window and mode.

    Parameters:
    - fama_french_data (pd.DataFrame): The DataFrame containing Fama-French factors, with a 'Date' column.
    - estimation_window (Tuple[int, int]): The window of days around the event_date to include.
    - event_date (datetime): The date of the event.
    - mode (str): The mode to specify which columns to use. Defaults to "market_model".

    Returns:
    - np.ndarray: The constructed X matrix.
    """
    # Calculate start and end dates based on the event date and estimation window
    start_date = event_date + pd.Timedelta(days=estimation_window[0])
    end_date = event_date + pd.Timedelta(days=estimation_window[1])

    # Filter the DataFrame for the estimation window
    estimation_data = fama_french_data[(fama_french_data['Date'] >= start_date) & (fama_french_data['Date'] <= end_date)]

    # Construct the X matrix based on the specified mode
    if mode == "market_model":
        # Include an intercept column (1s) and the 'Mkt' column
        X_matrix = np.column_stack((np.ones(len(estimation_data)), estimation_data['Mkt'].values))

    elif mode == 'three_factor_model':
        # Include an intercept column (1s) and the 'Mkt', 'SMB', and 'HML' columns
        X_matrix = np.column_stack((np.ones(len(estimation_data)), estimation_data['Mkt'].values, estimation_data['SMB'].values, estimation_data['HML'].values))

    elif mode == "constant_model":
        # Include an intercept column (1s)
        X_matrix = np.ones((len(estimation_data), 1))

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return X_matrix


def create_Y_matrix(stock_data: pd.DataFrame, event_date: datetime, event_window: Tuple[int, int]) -> np.ndarray:
    """
    Creates a Y matrix based on the specified event window, using the 'Daily_Return' column.

    Parameters:
    - stock_data (pd.DataFrame): DataFrame containing stock data with a 'Date' column and a 'Daily_Return' column.
    - event_date (datetime): The date of the event.
    - event_window (Tuple[int, int]): The range of days relative to the event_date to include in the matrix.

    Returns:
    - np.ndarray: The Y matrix containing 'Daily_Return' values within the event window.
    """
    # Calculate start and end dates based on the event date and event window
    start_date = event_date + pd.Timedelta(days=event_window[0])
    end_date = event_date + pd.Timedelta(days=event_window[1])

    # Filter the DataFrame for the event window
    event_data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]

    # Explicitly select the 'Daily_Return' column and reshape it into a 2D numpy array
    Y_matrix = event_data['Daily_Return'].values

    return Y_matrix


def create_X_star_matrix(event_date: datetime, event_window: Tuple[int, int], factor_data: pd.DataFrame, model_type: str = "market_model") -> np.ndarray:
    """
    Creates an X* matrix based on the specified event window, including an intercept and the 'Mkt' column.

    Parameters:
    - event_date (datetime): The date of the event.
    - event_window (Tuple[int, int]): The range of days relative to the event_date to include in the matrix.
    - factor_data (pd.DataFrame): DataFrame containing factor data with a 'Date' column and 'Mkt' column.

    Returns:
    - np.ndarray: The X* matrix containing an intercept and 'Mkt' values within the event window.
    """
    # Calculate start and end dates based on the event date and event window
    start_date = event_date + pd.Timedelta(days=event_window[0])
    end_date = event_date + pd.Timedelta(days=event_window[1])

    # Filter the DataFrame for the event window
    event_data = factor_data[(factor_data['Date'] >= start_date) & (factor_data['Date'] <= end_date)]

    # Construct the X* matrix with an intercept (1s) and the 'Mkt' column
    if model_type == "market_model":
        X_star_matrix = np.column_stack((np.ones(len(event_data)), event_data['Mkt'].values))

    elif model_type == "three_factor_model":
        X_star_matrix = np.column_stack((np.ones(len(event_data)), event_data['Mkt'].values, event_data['SMB'].values, event_data['HML'].values))

    elif model_type == "constant_model":
        X_star_matrix =  X_star_matrix = np.ones(len(event_data)).reshape(-1, 1) 

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return X_star_matrix

def create_Y_star_matrix(event_date: datetime, event_window: Tuple[int, int], stock_data: pd.DataFrame) -> np.ndarray:
    """
    Creates a Y* matrix based on the specified event window, using the 'Daily_Return' column.

    Parameters:
    - event_date (datetime): The date of the event.
    - event_window (Tuple[int, int]): The range of days relative to the event_date to include in the matrix.
    - stock_data (pd.DataFrame): DataFrame containing stock data with a 'Date' column and a 'Daily_Return' column.

    Returns:
    - np.ndarray: The Y* matrix containing 'Daily_Return' values within the event window.
    """
    # Calculate start and end dates based on the event date and event window
    start_date = event_date + pd.Timedelta(days=event_window[0])
    end_date = event_date + pd.Timedelta(days=event_window[1])

    # Filter the DataFrame for the event window
    event_data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]

    # Explicitly select the 'Daily_Return' column and reshape it into a 2D numpy array
    Y_star_matrix = event_data['Daily_Return'].values

    return Y_star_matrix


def estimate_return_model(X: np.ndarray, Y: np.ndarray, model_type: str = "market_model"):
    """As output, we need the model and the estimation for residual variance."""

    model = sm.OLS(Y, X).fit()
    residuals = model.resid

    if model_type == "market_model":

        residual_variance_estimate = np.dot(residuals, residuals) / (len(Y) - 2)

    elif model_type == "three_factor_model":

        residual_variance_estimate = np.dot(residuals, residuals) / (len(Y) - 4)

    elif model_type == "constant_model":

        residual_variance_estimate = np.dot(residuals, residuals) / (len(Y) - 1)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, residual_variance_estimate


def estimate_normal_abnormal_returns(model: np.ndarray, X_star: np.ndarray, Y_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates normal returns and abnormal returns based on a trained model and new data.

    Parameters:
    - model (np.ndarray): The trained OLS model.
    - X_star (np.ndarray): The input data for the event window to predict normal returns.
    - Y_star (np.ndarray): The actual returns during the event window.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: The normal returns and abnormal returns (residuals).
    """

    # Use the trained model to predict normal returns for X_star
    normal_returns = model.predict(X_star)

    # Calculate abnormal returns (residuals) as the difference between Y_star and the predicted normal returns
    abnormal_returns = Y_star - normal_returns

    return normal_returns, abnormal_returns


def estimate_variance_of_abnormal_returns(X: np.ndarray, Y: np.ndarray, X_star: np.ndarray, Y_star: np.ndarray, residual_variance_estimation: float) -> np.ndarray:
    """This function implements the estimator from Campbell, Lo, and MacKinlay (1997, p. 159)."""

    variance = np.eye(len(Y_star)) * residual_variance_estimation + X_star @ np.linalg.inv(X.T @ X) @ X_star.T * residual_variance_estimation

    return  variance


def calculate_cumulative_abnormal_returns(abnormal_returns: np.ndarray, time_period: int) -> np.ndarray:
    """This function calculates the cumulative abnormal returns. Time period is the number of days in the event window. The first day is day 0."""

    car = np.sum(abnormal_returns[:time_period+1])

    return car


def calculate_variance_of_cumulative_abnormal_returns(variance_of_abnormal_returns: np.ndarray) -> np.ndarray:
    """This function calculates the variance of the cumulative abnormal returns."""

    length = len(variance_of_abnormal_returns)

    # Calculate the product
    var_car = np.ones(length) @ variance_of_abnormal_returns @ np.ones(length).T

    return var_car


def create_event_study_output(
    event_date: datetime,
    stock_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    normal_returns: np.ndarray,
    abnormal_returns: np.ndarray,
    variance_of_abnormal_returns: np.ndarray,
    event_window: Tuple[int, int],
    historical_days: int = 5,
    model_type: str = "market_model"
) -> pd.DataFrame:
    """
    Creates a DataFrame containing event study results for each day starting from the event date.

    Parameters:
    - event_date: The date of the event
    - stock_data: DataFrame containing actual stock returns
    - factor_data: DataFrame containing market returns
    - normal_returns: Array of estimated normal returns
    - abnormal_returns: Array of calculated abnormal returns
    - variance_of_abnormal_returns: Variance-covariance matrix of abnormal returns
    - event_window: Tuple defining the event window (start_day, end_day)
    - historical_days: Number of previous days to include before the event window

    Returns:
    - pd.DataFrame containing daily event study metrics
    """
    # Ensure event_date is a Timestamp
    event_date = pd.to_datetime(event_date)

    # Calculate start and end dates including historical days
    start_date = event_date + pd.Timedelta(days=event_window[0] - historical_days)
    end_date = event_date + pd.Timedelta(days=event_window[1])

    # Filter stock_data to the date range and reset index
    date_mask = (stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)
    stock_data_filtered = stock_data.loc[date_mask].copy()
    stock_data_filtered.reset_index(drop=True, inplace=True)

    # Calculate Relative_Day
    stock_data_filtered['Relative_Day'] = (stock_data_filtered['Date'] - event_date).dt.days

    # Merge with factor_data
    stock_data_filtered = stock_data_filtered.merge(
        factor_data[['Date', 'Mkt']].rename(columns={'Mkt': 'Market_Return'}),
        on='Date',
        how='left'
    )

    # Initialize columns with NaN
    stock_data_filtered['Normal_Return'] = np.nan
    stock_data_filtered['Abnormal_Return'] = np.nan
    stock_data_filtered['Variance_Abnormal_Return'] = np.nan
    stock_data_filtered['CAR'] = np.nan
    stock_data_filtered['Variance_CAR'] = np.nan

    # Define event window mask
    event_window_mask = (stock_data_filtered['Relative_Day'] >= event_window[0]) & \
                        (stock_data_filtered['Relative_Day'] <= event_window[1])

    # Get indices for the event window
    event_window_indices = stock_data_filtered.index[event_window_mask]

    # Check lengths to prevent assignment errors
    num_event_days = len(event_window_indices)
    if num_event_days != len(normal_returns):
        raise ValueError("Length of normal_returns does not match the number of days in the event window.")
    if num_event_days != len(abnormal_returns):
        raise ValueError("Length of abnormal_returns does not match the number of days in the event window.")

    # Assign normal and abnormal returns
    stock_data_filtered.loc[event_window_indices, 'Normal_Return'] = normal_returns
    stock_data_filtered.loc[event_window_indices, 'Abnormal_Return'] = abnormal_returns
    stock_data_filtered.loc[event_window_indices, 'Variance_Abnormal_Return'] = np.diag(variance_of_abnormal_returns)

    # Calculate CAR and Variance_CAR
    stock_data_filtered.loc[event_window_indices, 'CAR'] = stock_data_filtered.loc[event_window_indices, 'Abnormal_Return'].cumsum()
    stock_data_filtered.loc[event_window_indices, 'Variance_CAR'] = stock_data_filtered.loc[event_window_indices, 'Variance_Abnormal_Return'].cumsum()

    # Calculate degrees of freedom
    if model_type == "market_model":
        degrees_of_freedom = len(abnormal_returns) - 2  # Adjust based on model parameters
    elif model_type == "constant_model":
        degrees_of_freedom = len(abnormal_returns) - 1
    elif model_type == "three_factor_model":
        degrees_of_freedom = len(abnormal_returns) - 4
    if degrees_of_freedom <= 0:
        print("Degrees of freedom must be positive.")

    # Critical t-value for 95% confidence interval
    t_critical = t.ppf(0.975, degrees_of_freedom)

    # Calculate statistics
    std_error = np.sqrt(stock_data_filtered['Variance_CAR'])
    stock_data_filtered['t_statistic'] = stock_data_filtered['CAR'] / std_error
    stock_data_filtered['p_value'] = 2 * (1 - t.cdf(np.abs(stock_data_filtered['t_statistic']), degrees_of_freedom))
    stock_data_filtered['CI_upper_bound_95'] = stock_data_filtered['CAR'] + t_critical * std_error
    stock_data_filtered['CI_lower_bound_95'] = stock_data_filtered['CAR'] - t_critical * std_error

    # Select and order columns
    result_columns = [
        'Date', 'Relative_Day', 'Daily_Return', 'Normal_Return',
        'Market_Return', 'Abnormal_Return', 'CAR',
        'Variance_Abnormal_Return', 'Variance_CAR',
        'CI_lower_bound_95', 'CI_upper_bound_95',
        't_statistic', 'p_value'
    ]


    return stock_data_filtered[result_columns]

    




def run_event_study(
    ticker: str,
    event_date: datetime,
    estimation_window: Tuple[int, int] = (-120, -11),
    event_window: Tuple[int, int] = (-10, 10),
    historical_days: int = 10,
    model_type: str = "market_model"
) -> pd.DataFrame:
    """
    Runs a complete event study analysis for a given ticker and event date.
    
    Parameters:
    - ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
    - event_date: The date of the event
    - estimation_window: Tuple defining the estimation window relative to event date (default: -120 to -11 days)
    - event_window: Tuple defining the event window relative to event date (default: -10 to +10 days)
    - historical_days: Number of days before event window to include in output (default: 5)
    - model_type: Type of model to use for normal returns estimation (default: "market_model")
    
    Returns:
    - pd.DataFrame containing the event study results
    """
    # Convert string date to datetime if necessary
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date)
    
    # Calculate the start and end dates for data fetching
    start_date = event_date + pd.Timedelta(days=min(estimation_window[0], event_window[0]) - historical_days)
    end_date = event_date + pd.Timedelta(days=max(estimation_window[1], event_window[1]) + historical_days)
    
    # Fetch stock data using yfinance
    stock = yf.download(ticker, start=start_date, end=end_date)
    stock_data = clean_stock_returns(stock)
    
    # Check if the event date exists in the data (i.e., if it's a trading day)
    event_day_exists = any((stock_data['Date'].dt.date == event_date.date()))
    if not event_day_exists:
        raise ValueError(f"No market data available for {event_date.date()}. Please specify a day when the market is open to run an event study.")
    
    # Get and clean Fama-French factors
    ff_raw = get_fama_french_daily_factors()
    ff_data = clean_fama_french_factor_data(ff_raw)
    
    # Create matrices for estimation
    X = create_X_matrix(
        fama_french_data=ff_data,
        estimation_window=estimation_window,
        event_date=event_date,
        mode=model_type
    )
    
    Y = create_Y_matrix(
        stock_data=stock_data,
        event_date=event_date,
        event_window=estimation_window
    )
    
    # Create matrices for prediction
    X_star = create_X_star_matrix(
        event_date=event_date,
        event_window=event_window,
        factor_data=ff_data, 
        model_type=model_type
    )
    
    Y_star = create_Y_star_matrix(
        event_date=event_date,
        event_window=event_window,
        stock_data=stock_data
    )
    
    # Estimate model and get residual variance
    model, residual_variance = estimate_return_model(
        X=X,
        Y=Y,
        model_type=model_type
    )
    
    # Calculate normal and abnormal returns
    normal_returns, abnormal_returns = estimate_normal_abnormal_returns(
        model=model,
        X_star=X_star,
        Y_star=Y_star
    )
    
    # Calculate variance of abnormal returns
    variance_abnormal_returns = estimate_variance_of_abnormal_returns(
        X=X,
        Y=Y,
        X_star=X_star,
        Y_star=Y_star,
        residual_variance_estimation=residual_variance
    )
    
    # Create final output
    output = create_event_study_output(
        event_date=event_date,
        stock_data=stock_data,
        factor_data=ff_data,
        normal_returns=normal_returns,
        abnormal_returns=abnormal_returns,
        variance_of_abnormal_returns=variance_abnormal_returns,
        event_window=event_window,
        historical_days=historical_days
    )
    
    return output


def plot_CAR_over_time(event_study_results,
                       days_before_event: int = 10,
                       days_after_event: int = 20,
                       plot_colors = ["#3c5488", "#e64b35", "#4dbbd5", "#00a087", "#f39b7f", "#000000"]):
    """
    Plot Cumulative Abnormal Returns (CAR) over time with confidence intervals using Seaborn.
    """
    # Infer event date from the data (where Relative_Day = 0)
    event_date = event_study_results[event_study_results['Relative_Day'] == 0]['Date'].iloc[0]

    # Filter the DataFrame for the specified window
    event_study_results = event_study_results[event_study_results['Relative_Day'] >= -days_before_event]
    event_study_results = event_study_results[event_study_results['Relative_Day'] <= days_after_event]
    
    # Clear any existing plots
    plt.clf()
    
    # Set the white theme and LaTeX font with larger font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })
    sns.set_theme(style="white")

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the data using plt instead of sns
    ax.plot(event_study_results['Date'], event_study_results['Daily_Return'], 
            label='Observed Return', color=plot_colors[5], linestyle=':')
    ax.plot(event_study_results['Date'], event_study_results['Normal_Return'], 
            label='Normal Return', color=plot_colors[0], linestyle='--')
    ax.plot(event_study_results['Date'], event_study_results['Abnormal_Return'], 
            label='Abnormal Return', color=plot_colors[3])
    ax.plot(event_study_results['Date'], event_study_results['CAR'], 
            label='Cumulative Abnormal Return', color=plot_colors[2])

    # Add confidence intervals
    ax.fill_between(event_study_results['Date'], 
                   event_study_results['CI_lower_bound_95'], 
                   event_study_results['CI_upper_bound_95'], 
                   color='grey', alpha=0.1)

    # Add the event date line and label
    ax.axvline(x=event_date, color='grey', linestyle='-', alpha=0.5)

    # Add a grey dotted line at x=0
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)

    # Set the labels
    ax.set_xlabel('')
    ax.set_ylabel('Value')

    # Calculate y-axis limits based on data
    all_values = pd.concat([
        event_study_results['Normal_Return'],
        event_study_results['Abnormal_Return'],
        event_study_results['CAR'], 
        event_study_results['Daily_Return'],
    ])
    max_val = all_values.max()
    min_val = all_values.min()
    ax.set_ylim(min_val * 1.1, max_val * 1.1)

    # Remove the top and right spines
    sns.despine()

    # Show the legend with larger font
    ax.legend(prop={'size': 14})

    # Format date axis to prevent overlapping
    fig.autofmt_xdate()  # Rotate and align the tick labels
    
    # Additional date formatting options
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.close(fig)

    return fig
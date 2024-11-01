from easy_event_studies import get_fama_french_daily_factors, create_X_matrix, create_Y_matrix, create_X_star_matrix, create_Y_star_matrix, estimate_return_model, estimate_normal_abnormal_returns, estimate_variance_of_abnormal_returns, create_event_study_output, calculate_cumulative_abnormal_returns, calculate_variance_of_cumulative_abnormal_returns

import requests
import zipfile
import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np
import pdb
from scipy.stats import t


@pytest.fixture
def mock_dataframe_fama_french_factors():
    data = {
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'Mkt-RF': [0.01, 0.02, 0.015, 0.01, 0.015],
        'SMB': [0.001, 0.002, 0.0015, 0.001, 0.0015],
        'Mkt': [5, 5, 7, 3, 5],
        'HML': [0.005, 0.004, 0.006, 0.005, 0.006],
        'RF': [0.0001, 0.0002, 0.0001, 0.0001, 0.0001],
    }
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])  # Ensuring Date column is datetime if needed
    return df

@pytest.fixture
def mock_dataframe_stock_data():
    data = {
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'Daily_Return': [6, 8, 10, 12, 16],
    }
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def test_create_X_matrix(mock_dataframe_fama_french_factors):

    estimation_window = [-3, -1]
    event_date = pd.to_datetime('2024-01-04')

    expected_result = np.array([[1, 5],
                                [1, 5],
                                [1, 7]])

    actual_result = create_X_matrix(mock_dataframe_fama_french_factors, estimation_window, event_date, mode="market_model")

    assert np.allclose(actual_result, expected_result, rtol=1e-15, atol=1e-15)


def test_create_Y_matrix(mock_dataframe_stock_data):

    event_date = pd.to_datetime('2024-01-04')
    event_window = [-3, -1]

    expected_result = np.array([6, 8, 10])

    actual_result = create_Y_matrix(mock_dataframe_stock_data, event_date, event_window)

    assert np.allclose(actual_result, expected_result, rtol=1e-15, atol=1e-15)



def test_create_X_star_matrix(mock_dataframe_fama_french_factors):

    event_date = pd.to_datetime('2024-01-04')
    event_window = [0,1]

    expected_result = np.array([[1, 3],
                                [1, 5]])

    actual_result = create_X_star_matrix(event_date, event_window, mock_dataframe_fama_french_factors)

    assert np.allclose(actual_result, expected_result, rtol=1e-15, atol=1e-15)



def test_create_Y_star_matrix(mock_dataframe_stock_data):

    event_date = pd.to_datetime('2024-01-04')
    event_window = [0,1]

    expected_result = np.array([12, 16])

    actual_result = create_Y_star_matrix(event_date, event_window, mock_dataframe_stock_data)

    assert np.allclose(actual_result, expected_result, rtol=1e-15, atol=1e-15)



def test_estimate_return_model(mock_dataframe_fama_french_factors, mock_dataframe_stock_data):

    X = create_X_matrix(mock_dataframe_fama_french_factors, [-3, -1], pd.to_datetime('2024-01-04'), mode="market_model")
    Y = create_Y_matrix(mock_dataframe_stock_data, pd.to_datetime('2024-01-04'), [-3, -1])

    model, residual_variance_estimate = estimate_return_model(X, Y, model_type="market_model")

    expected_params = np.array([-0.5, 1.5])  # Example: Intercept=1.0, Slope=0.5

     # Check that the model's parameters are close to the expected values
    assert np.allclose(model.params, expected_params, rtol=1e-5, atol=1e-5), \
        f"Expected parameters {expected_params}, but got {model.params}"

    assert np.isclose(residual_variance_estimate, 2.0), f"Expected residual variance to be 2.0, but got {residual_variance_estimate}"



def test_estimate_normal_returns(mock_dataframe_fama_french_factors, mock_dataframe_stock_data):

    X = create_X_matrix(mock_dataframe_fama_french_factors, [-3, -1], pd.to_datetime('2024-01-04'), mode="market_model")
    Y = create_Y_matrix(mock_dataframe_stock_data, pd.to_datetime('2024-01-04'), [-3, -1])

    X_star = create_X_star_matrix(pd.to_datetime('2024-01-04'), [0,1], mock_dataframe_fama_french_factors)
    Y_star = create_Y_star_matrix(pd.to_datetime('2024-01-04'), [0,1], mock_dataframe_stock_data)

    model, _ = estimate_return_model(X, Y, model_type="market_model")

    expected_normal_returns = np.array([4,7])
    expected_abnormal_returns = np.array([8,9])

    actual_normal_returns, actual_abnormal_returns = estimate_normal_abnormal_returns(model, X_star, Y_star)


    assert np.allclose(actual_normal_returns, expected_normal_returns, rtol=1e-15, atol=1e-15)
    assert np.allclose(actual_abnormal_returns, expected_abnormal_returns, rtol=1e-15, atol=1e-15)


def test_estimate_variance_of_abnormal_returns(mock_dataframe_fama_french_factors, mock_dataframe_stock_data):

    Y = create_Y_matrix(mock_dataframe_stock_data, pd.to_datetime('2024-01-04'), [-3, -1])
    X = create_X_matrix(mock_dataframe_fama_french_factors, [-3, -1], pd.to_datetime('2024-01-04'), mode="market_model")
    X_star = create_X_star_matrix(pd.to_datetime('2024-01-04'), [0,1], mock_dataframe_fama_french_factors)
    Y_star = create_Y_star_matrix(pd.to_datetime('2024-01-04'), [0,1], mock_dataframe_stock_data)

    model, _ = estimate_return_model(X, Y, model_type="market_model")

    residual_variance_estimate = 2.0

    actual_variance = estimate_variance_of_abnormal_returns(X, Y, X_star, Y_star, residual_variance_estimate)

    expected_variance = np.array([[8, 2], [2, 3]])

    assert np.allclose(actual_variance, expected_variance, rtol=1e-10, atol=1e-10)



def test_calculate_cumulative_abnormal_returns():

    abnormal_returns = np.array([8,9])
    time_period = 1

    actual_car = calculate_cumulative_abnormal_returns(abnormal_returns, time_period)

    expected_car = 17

    assert np.isclose(actual_car, expected_car, rtol=1e-15, atol=1e-15)


def test_calculate_variance_of_cumulative_abnormal_returns():

    variance_of_abnormal_returns = np.array([[8, 2], [2, 3]])

    actual_var_car = calculate_variance_of_cumulative_abnormal_returns(variance_of_abnormal_returns)

    expected_var_car = 15

    assert np.isclose(actual_var_car, expected_var_car, rtol=1e-15, atol=1e-15)


def test_create_event_study_output(mock_dataframe_stock_data, mock_dataframe_fama_french_factors):
    # Test parameters
    event_date = pd.to_datetime('2024-01-04')
    historical_days = 3
    event_window = [0, 1]
    
    # Values from previous tests
    normal_returns = np.array([4, 7])
    abnormal_returns = np.array([8, 9])
    variance_of_abnormal_returns = np.array([[8, 2], [2, 3]])
    
    # Calculate expected statistical values
    degrees_of_freedom = len(abnormal_returns) - 2
    t_critical = t.ppf(0.975, degrees_of_freedom)
    
    # Expected t-statistics and p-values
    t_stat_0 = 8.0 / np.sqrt(8.0)  # CAR / sqrt(Variance_CAR) for day 0
    t_stat_1 = 17.0 / np.sqrt(11.0)  # CAR / sqrt(Variance_CAR) for day 1
    p_val_0 = 2 * (1 - t.cdf(abs(t_stat_0), degrees_of_freedom))
    p_val_1 = 2 * (1 - t.cdf(abs(t_stat_1), degrees_of_freedom))
    
    # Create expected DataFrame
    expected_data = {
        'Date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),
        'Relative_Day': [-3, -2, -1, 0, 1],
        'Daily_Return': [6.0, 8.0, 10.0, 12.0, 16.0],
        'Normal_Return': [pd.NA, pd.NA, pd.NA, 4.0, 7.0],
        'Market_Return': [5.0, 5.0, 7.0, 3.0, 5.0],
        'Abnormal_Return': [pd.NA, pd.NA, pd.NA, 8.0, 9.0],
        'CAR': [pd.NA, pd.NA, pd.NA, 8.0, 17.0],
        'Variance_Abnormal_Return': [pd.NA, pd.NA, pd.NA, 8.0, 3.0],
        'Variance_CAR': [pd.NA, pd.NA, pd.NA, 8.0, 11.0],
        'CI_lower_bound_95': [pd.NA, pd.NA, pd.NA, 
                            8.0 - t_critical * np.sqrt(8.0),
                            17.0 - t_critical * np.sqrt(11.0)],
        'CI_upper_bound_95': [pd.NA, pd.NA, pd.NA,
                            8.0 + t_critical * np.sqrt(8.0),
                            17.0 + t_critical * np.sqrt(11.0)],
        't_statistic': [pd.NA, pd.NA, pd.NA, t_stat_0, t_stat_1],
        'p_value': [pd.NA, pd.NA, pd.NA, p_val_0, p_val_1]
    }
    expected_result = pd.DataFrame(expected_data)
    
    # Get actual result
    actual_result = create_event_study_output(
        event_date=event_date,
        stock_data=mock_dataframe_stock_data,
        factor_data=mock_dataframe_fama_french_factors,
        normal_returns=normal_returns,
        abnormal_returns=abnormal_returns,
        variance_of_abnormal_returns=variance_of_abnormal_returns,
        event_window=event_window,
        historical_days=historical_days
    )
    
    # Assert equality
    pd.testing.assert_frame_equal(
        actual_result,
        expected_result,
        check_dtype=False  # Since we're mixing floats and NA values
    )


    
# EasyEventStudies

This package makes it easy to run event studies on financial assets. In contrast to other packages, it is easy to use, correct and allows for three different models of normal returns estimation. All calculations are based on the standard reference in the literature: *The Econometrics of Financial Markets; John Y. Campbell, Andrew W. Lo and Craig MacKinlay*
## Quick Start Guide

Install the package using `pip`:

```bash
pip install pip install EasyEventStudies==1.0
```

The `run_event_study` function runs a complete event study analysis for a financial asset and an event date. The package takes financial data from Yahoo Finance, so search for the ticker symbol of the asset you are interested on their website [here](https://finance.yahoo.com/). 
As an example, we will run an event study on Boeing Co. (ticker: BA) on 8th of March, 2019. At this time, the Boing 737 Max was involved in two crashes and the FAA was about to ground the fleet.

```python

results = run_event_study(
    ticker='BA',
    event_date='2019-03-08',
    estimation_window=[-250, 0],
    event_window=[0, 20],
    model_type="market_model"
)

plot_CAR_over_time(results)
```
<img src="./images/boing_example.png" width="70%"/>

The event study shows that the failure of the Boing 737 Max was associated with a drop in the stock price of about 17 Percent in the first 20 days after the event.


## More Detailed Documentation

1. The main function is `run_event_study`. It returns a pandas DataFrame with the results of the event study.

```python
def run_event_study(
    ticker: str,
    event_date: str,
    estimation_window: Tuple[int, int],
    event_window: Tuple[int, int],
    historical_days: int = 10,
    model_type: str = "market_model"
):
```
- **ticker**: The ticker symbol from Yahoo Finance as a string (e.g., 'BA' for Boeing)
- **event_date**: The date of the event in 'YYYY-MM-DD' format (e.g., '2019-03-08')
- **estimation_window**: A tuple of two integers defining the time period used to estimate normal returns. The first number is the start of the window, the second is the end. For example, [-250, -1] uses the previous year's data.
- **event_window**: A tuple of two integers defining the period for calculating cumulative abnormal returns. The first number is the start, the second is the end. For example, [0, 10] analyzes the 10 days following the event.
- **historical_days**: Number of days before the event window to include in the output. This allows for plotting pre-event stock returns. Defaults to 10.
- **model_type**: The model to use for estimating normal returns. Options are:
  - "market_model": Estimates returns as a function of market returns
  - "constant_model": Assumes constant normal returns
  - "three_factor_model": Uses the Fama-French three-factor model

2. The `plot_CAR_over_time` function plots the cumulative abnormal returns over time. It takes the results of the `run_event_study` function as an input and plots the cumulative abnormal returns over time.

```python
def plot_CAR_over_time(event_study_results,
                       days_before_event: int = 10, 
                       days_after_event: int = 10
                       ):
```

- **event_study_results**: The results of the `run_event_study` function.
- **days_before_event**: Number of days before the event window to include in the plot. Defaults to 10.
- **days_after_event**: Number of days after the event window to include in the plot. Defaults to 10.


# Citation
If you use this package in your work, please cite it as:

```
Roever, Nicolas (2024). EasyEventStudies: A Python Package for Event Studies. https://github.com/NicolasRoever/EasyEventStudies

```

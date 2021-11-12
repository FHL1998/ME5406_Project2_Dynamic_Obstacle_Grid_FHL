import pandas


def mean_smooth(input_data, smoothness):
    return pandas.Series(list(input_data)).rolling(smoothness, min_periods=5).mean()
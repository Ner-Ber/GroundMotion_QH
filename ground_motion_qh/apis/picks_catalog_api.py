# An extension to pandas df for earthquake catalogs

import functools
from typing import Optional
import matplotlib as mpl
import pandas as pd
import numpy as np
import cartopy
from cartopy import crs
import matplotlib.pyplot as plt
from pandas.api.extensions import register_dataframe_accessor


# ----- Default Parameters -----
DEF_FORECAST_WIN_SEC = 10*60*24*7  # 1 week in seconds


def load_picks_catalog(
    file_path: str,
    read_csv_kwargs: Optional[dict] = None
) -> pd.DataFrame:
  """
  Load a picks catalog from a CSV file into a pandas DataFrame.
  """
  picks_df = pd.read_csv(file_path, index_col=0, **read_csv_kwargs)
  picks_df['epoch_time'] = pd.to_datetime(picks_df['time']).astype(np.int64) // 10**9
  picks_df['amp_epoch_time'] = pd.to_datetime(
      picks_df['amp_time1']).astype(np.int64) // 10**9
  picks_df['p_epoch_time'] = pd.to_datetime(
      picks_df['p_time']).astype(np.int64) // 10**9
  picks_df['s_epoch_time'] = pd.to_datetime(
      picks_df['s_time']).astype(np.int64) // 10**9
  picks_df = picks_df.sort_values(by='epoch_time')
  return picks_df


@register_dataframe_accessor("pick_catalog")
class CustomDataFrameTools:
  """
  """

  def __init__(self, pandas_obj):
    # The 'pandas_obj' is the DataFrame instance this accessor is attached to.
    self._df = pandas_obj

  # ----- Data Preparation Methods -----
  DEF_FORECAST_WIN_SEC = 10*60*24*7  # 1 week in seconds

  def create_pairs(
      self,
      trigger_column='amp',
      forecast_column='amp',
      trigger_threshold=-np.inf,
      threshold_column=None,
      sequential=False,
      trigger_time_column='p_epoch_time',
      forecast_time_column='amp_epoch_time',
      mid_buffer=60,
      forecast_window_time=DEF_FORECAST_WIN_SEC
  ):

    col_values = self._df[trigger_column].values
    trigger_values = col_values[:-1]
    trigger_times = self._df[trigger_time_column].values[:-1]
    if threshold_column is None:
      threshold_column = trigger_column
    threshold_values = self._df[threshold_column].values[:-1]
    trigger_boolean = threshold_values > trigger_threshold

    forecast_values = []
    time_differences_sec = []
    for t in trigger_times[trigger_boolean]:
      if sequential:
        following_trigger_bool = self._df[forecast_time_column].values > t
        forecast_val = self._df[forecast_column].values[following_trigger_bool][0]
        time_diff = self._df[forecast_time_column].values[following_trigger_bool][0] - t
      else:
        # print(
          # f"t: {t}, mid_buffer: {mid_buffer}, forecast_window_time: {forecast_window_time}")
        first_index = np.searchsorted(
            self._df[forecast_time_column].values, t + mid_buffer, side='left')
        last_index = np.searchsorted(
            self._df[forecast_time_column].values, t + mid_buffer + forecast_window_time, side='left')
        time_slice = slice(first_index, last_index)
        # print(
        #     f"slice: {time_slice}, first_index: {first_index}, last_index: {last_index}")
        if last_index > first_index:
          forecast_val = np.max(self._df[forecast_column].values[time_slice])
          forecast_val_slice_idx = np.argmax(
              self._df[forecast_column].values[time_slice])
          time_diff = self._df[forecast_time_column].values[time_slice][forecast_val_slice_idx] - t
        else:
          forecast_val = np.nan
          time_diff = np.nan
      forecast_values.append(forecast_val)
      time_differences_sec.append(time_diff)
    pairs_array = np.array([trigger_values[trigger_boolean], forecast_values]).T
    time_differences_sec = np.array(time_differences_sec)

    return pairs_array, time_differences_sec

  # ----- Plotting Methods -----

  def plot_time_series(
      self,
      column: str = "amp",
      time_column="amp_epoch_time",
      ax=None,
      scatter_kwargs: Optional[dict] = {},
  ) -> mpl.axes.Axes:
    """
    Plots a time series of a given column in a dataframe.
    """
    if ax is None:
      fig, ax = plt.subplots()
    else:
      fig = ax.figure
    fig, ax = plt.subplots(figsize=(10, 6))

    kwargs_combined = {
        "s": 3.8,
        "c": 'k',
        "alpha": 0.4,
        "marker": "."
    }
    if scatter_kwargs is not None:
      kwargs_combined.update(scatter_kwargs)
    _ = ax.scatter(
        self._df[time_column],
        self._df[column],
        **kwargs_combined
    )

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel(column)
    ax.set_yscale("log")
    return fig, ax

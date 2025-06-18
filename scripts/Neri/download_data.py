import datetime
from typing import Sequence
from absl import app
from absl import flags
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
from matplotlib import pyplot as plt
import ground_motion_qh
from ground_motion_qh.earthquake import EarthquakeCatalog
from ground_motion_qh.get_waveforms import get_stream_multiple_stations, raw_stream_to_amplitude_and_times
import os
from pathlib import Path
from pprint import pprint
import pickle
from obspy.taup import TauPyModel
model = TauPyModel(model="iasp91")

BASE_DIR = Path(ground_motion_qh.__file__).parent.parent

# # check if the folder already exists in the raw data folder
# DATA_DIR = BASE_DIR / "data" / "raw_data" / DOWNLOAD_NAME
# if DATA_DIR.exists():
#   print(f"Folder {DOWNLOAD_NAME} already exists in the raw data folder.")

# DEFAULTS
_DEF_STATIONS = "SND"
_DEF_NETWORK = "AZ"
_DEF_ORG = "IRIS"
_DEF_NUM_TRIES = 2
_DEF_MID_BUFFER = 20
_DEF_FORECAST_TIME_WINDOW = 4*60*60
_DEF_EVENT_TIME_WINDOW = 30
_DEF_SHIFT_TIMES = True  # @keliankaz what is this for?


_START_TIMES = flags.DEFINE_string(
    'start_times', None, 'String or path to file containing start times.'
)


_DATA_DIR_PARENT = flags.DEFINE_string(
    'data_dir', None, 'Parent folder for creating the saving dir.'
)

_DOWNLOAD_NAME = flags.DEFINE_string(
    'download_name', None, 'Name of the folder to save the data.'
)

_OVERRIDE_DOWNLOAD_NAME = flags.DEFINE_boolean(
    'override_download_name', False,
    'If True, the download name will be overridden if already exists.'
)

_NAME_REFIX = flags.DEFINE_string(
    'name_prefix', '',
    'Prefix to the download name. This is useful for mentioning the region for exmpl.'
)

_MID_BUFFER = flags.DEFINE_float(
    'mid_buffer', _DEF_MID_BUFFER,
    'Mid buffer time in seconds. This is the time between the end of the event time window and the start of the forecast time window.'
)

_FORECAST_TIME_WINDOW = flags.DEFINE_float(
    'forecast_time_window', _DEF_FORECAST_TIME_WINDOW,
    'Forecast time window in seconds. This is the time after the event time window.'
)

_EVENT_TIME_WINDOW = flags.DEFINE_float(
    'event_time_window', _DEF_EVENT_TIME_WINDOW,
    'Event time window in seconds. This is the time before the mid buffer.'
)

_SHIFT_TIMES = flags.DEFINE_boolean(
    'shift_times', _DEF_SHIFT_TIMES,
    'Shift times. If True, the start times will be shifted by the pre-buffer time.'
)  # @keliankaz what is this for?

_STNAME = flags.DEFINE_string('station', _DEF_STATIONS,
                              'Station name to download data for.')
_NETWORK = flags.DEFINE_string(
    'network', _DEF_NETWORK, 'Network name to download data for.')
_ORG = flags.DEFINE_string('org', _DEF_ORG, 'Organization name to download data for.')


def _create_download_name(
    stname: str,
    network: str,
    org: str,
    event_time_window: float,
    forecast_time_window: float,
    mid_buffer: float,
) -> str:
  """

  """
  if _DOWNLOAD_NAME.value is not None:
    download_name = _DOWNLOAD_NAME.value
  else:
    download_name = f"{stname}_{network}_{org}_event_{event_time_window}s_forecast_{forecast_time_window}s_mid_{mid_buffer}s"
  if len(_NAME_REFIX.value) > 0:
    download_name = f"{_NAME_REFIX.value}_{download_name}"
  download_name = download_name.replace(' ', '_')
  download_name = download_name.replace('.', '_')
  download_name = download_name.replace(',', '_')
  return download_name


def _parse_data_dir_parent(
    data_dir: str | None,
    download_name: str,
) -> Path:
  """
  Parse the data directory parent from the command line flag or use the default.
  """
  if data_dir is None:
    parsed_dir = BASE_DIR / "data" / "raw_data" / download_name
  else:
    parsed_dir = Path(data_dir / download_name)

  # If the dir already exists and override is false add date and time to its name
  if _OVERRIDE_DOWNLOAD_NAME.value is False and parsed_dir.exists():
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parsed_dir = parsed_dir.parent / f"{parsed_dir.name}_{date_str}"
  # If the dir already exists and override is true, we can just return the parsed_dir
  os.makedirs(parsed_dir, exist_ok=True)
  return parsed_dir


def format_string_from_utcdatetime(utc_datetime: UTCDateTime):
  """formats UTCDateTime to a string in the format:
  (YYYY, MM, DD, HH, MM, SS, f)

  THIS METHOD DOES NOT RETURN NANOSECONDS!!"""

  format_string = "(%Y, %m, %d, %H, %M, %S, %f)"
  return utc_datetime.strftime(format_string)


def start_times_to_file(
    start_times: Sequence[UTCDateTime],
    saving_path: str,
):
  """
  Save start times to a file.
  start_times example =
    [
      UTCDateTime(2023, 10, 1, 0, 0),
      UTCDateTime(2024, 3, 7, 15, 20, 55),
      UTCDateTime(2023, 10, 2),
      UTCDateTime(2024, 3, 8, 16, 21, 56),
      UTCDateTime(1990, 1, 3),
      UTCDateTime(1991, 4, 24, 16, 59, 40),
      UTCDateTime(1905, 11, 5, 12, 0, 3, 5),
      UTCDateTime(1915),
    ]
  """
  with open(saving_path, 'w') as f:
    for t1 in start_times:
      # format the UTCDateTime to a string
      t1_str = format_string_from_utcdatetime(t1)
      # write the time pair to the file
      f.write(f"{t1_str}\n")


def _parse_start_times(start_times: str):
  """
  Parse start times from a string or file.

  start_times: str
    If a path to a file, it should contain lines with time pairs in the format:
      (YYYY1, MM1, DD1, HH1, MM1, SS1)
      (YYYY2, MM2, DD2, HH2, MM2, SS2)
      (YYYY3, MM3, DD3, HH3, MM3, SS3)
      (YYYY4, MM4, DD4, HH4, MM4, SS4)
      ....

    If a string, it should contain time pairs in the format:
      "(YYYY1, MM1, DD1, HH1, MM1, SS1), (YYYY2, MM2, DD2, HH2, MM2, SS2), (YYYY3, MM3, DD3, HH3, MM3, SS3), (YYYY4, MM4, DD4, HH4, MM4, SS4), ..."
  """
  if os.path.isfile(start_times):
    return _parse_pairs_file(start_times)
  else:
    return _parse_times_string(start_times)


def _parse_pairs_file(file_path: str):
  with open(file_path, 'r') as f:
    lines = f.readlines()
  return [_parse_time_line_from_file(line.strip('\n')) for line in lines]


def _time_str_parts_to_utc_datetime(time_components_str):
  """Input is a string:
  'YYYY1, MM1, DD1, HH1, MM1, SS1'"""
  time = UTCDateTime(*map(int, time_components_str.strip().strip(',').split(',')))
  return time


def _parse_time_line_from_file(line: str):
  """
  Parse a single line containing a time pair from a file.
  The line should be in the format:
    (YYYY1, MM1, DD1, HH1, MM1, SS1)
  """
  # This should return str in format: 'YYYY1, MM1, DD1, HH1, MM1, SS1':
  parts = line.replace(' ', '').strip(')(')
  return _time_str_parts_to_utc_datetime(parts)


def _parse_times_string(time_pair_string: str):
  """
  Parse a string containing time pairs in the format:
    "(YYYY1, MM1, DD1, HH1, MM1, SS1), (YYYY2, MM2, DD2, HH2, MM2, SS2), (YYYY3, MM3, DD3, HH3, MM3, SS3), (YYYY4, MM4, DD4, HH4, MM4, SS4), ..."
  """
  # reformatted_str = time_pair_string.replace(
  #     ' ', '').replace('((', '[(').replace('))', ')]')
  reformatted_str = time_pair_string.replace(
      ' ', '').replace('[', '(').replace(']', ')')
  reformatted_str = reformatted_str.strip('[]').strip('()')
  time_strings = reformatted_str.split('),(')
  time_strings = [t.strip(' (),') for t in time_strings]
  times_utc = [_time_str_parts_to_utc_datetime(t) for t in time_strings]
  return times_utc


def create_and_save_metadata(
    start_times: Sequence[UTCDateTime],
    # stname=["SND"],
    # network="AZ",
    # org="IRIS",
    # latitude=33.5519,
    # longitude=-116.6129,
    # start_time='1982-10-01',
    number_of_tries=_DEF_NUM_TRIES,
    # starttime='1994-01-01',
    # endtime='2023-01-01',
    # latitude_range=[32, 35],
    # longitude_range=[-117.5, -115.5],
    # minimum_magnitude=4.0,
    mid_buffer=_DEF_MID_BUFFER,
    forecast_time_window=_DEF_FORECAST_TIME_WINDOW,
    event_time_window=_DEF_EVENT_TIME_WINDOW,
    shift_times=_DEF_SHIFT_TIMES,
):
  # MAKE SURE TO CHANGE STATION LOCATION ALONG WITH THE STATION NAME!
  station_metadata = dict(
      stname=_STNAME.value,
      network=_NETWORK.value,
      org=_ORG.value,
      latitude=None,
      longitude=None,
      # start_time=start_time,
      number_of_tries=number_of_tries,
  )

  # earthquake_metadata = {
  #     "starttime": starttime,
  #     "endtime": endtime,
  #     "latitude_range": latitude_range,
  #     "longitude_range": longitude_range,
  #     "minimum_magnitude": minimum_magnitude,
  # }

  # explanation of buffers:

  # (hypocenter) t1                                                                                            t2
  # t0 time shift    pre-buffer     event time window     mid buffer     forecast time window      post buffer
  # |------------||--------------||-------------------||--------------||---------- ... ---------||-------------|

  analysis_metadata = dict(
      mid_buffer=mid_buffer,
      forecast_time_window=forecast_time_window,
      event_time_window=event_time_window,
      shift_times=shift_times,  # @keliankaz what is this for?
      number_of_tries=number_of_tries,  # moved here from station metadata (@Ner-Ber)
  )

  analysis_metadata["pre_buffer"] = 3*0.05 * analysis_metadata["forecast_time_window"] + \
      analysis_metadata["event_time_window"] + analysis_metadata["mid_buffer"]
  analysis_metadata["post_buffer"] = 3*0.05 * analysis_metadata["forecast_time_window"] + \
      analysis_metadata["event_time_window"] + analysis_metadata["mid_buffer"]

  pprint(analysis_metadata, width=1)
  metadata = dict(
      # earthquake_metadata=earthquake_metadata,
      analysis_metadata=analysis_metadata,
      station_metadata=station_metadata,
  )

  download_name = _create_download_name(
      station_metadata["stname"],
      station_metadata["network"],
      station_metadata["org"],
      analysis_metadata["event_time_window"],
      analysis_metadata["forecast_time_window"],
      analysis_metadata["mid_buffer"],
  )

  download_dir = _parse_data_dir_parent(
      _DATA_DIR_PARENT.value, download_name
  )

  metadata_file_path = download_dir / "metadata.npy"
  if os.path.isfile(metadata_file_path):
    np.save(metadata_file_path, metadata)

  # Save start and end time pairs to a file
  start_times_file = download_dir / "start_times.txt"
  start_times_to_file(
      start_times, start_times_file
  )

  # to load the metadata
  # metadata = np.load(data_dir / "metadata.npy", allow_pickle=True).item()
  # return metadata, earthquake_metadata, analysis_metadata, station_metadata
  return metadata, analysis_metadata, download_dir


# earthquakes = EarthquakeCatalog(
#     filename=DATA_DIR / "local_catalog.csv",
#     kwargs=earthquake_metadata,
# )


def download_waveforms(
    start_times: Sequence[UTCDateTime],
    metadata: dict,
    saving_dir: Path,
):
  # earthquake_metadata = metadata["earthquake_metadata"]
  analysis_metadata = metadata["analysis_metadata"]
  station_metadata = metadata["station_metadata"]
  station_name = station_metadata["stname"]
  network = station_metadata["network"]
  org = station_metadata["org"]
  client = FDSN_Client(org)
  plus_time_range = [
      analysis_metadata["pre_buffer"] +
      analysis_metadata["event_time_window"] + analysis_metadata["mid_buffer"],
      analysis_metadata["pre_buffer"] + analysis_metadata["event_time_window"] +
      analysis_metadata["mid_buffer"] + analysis_metadata["forecast_time_window"]
  ]  # THIS NEEDS REVIEW

  a_max_minus = []
  a_max_plus = []

  for i, t0 in enumerate(start_times):
    # preprocess the waveforms
    i = 0
    # while i < station_metadata["number_of_tries"]:
    while i < _DEF_NUM_TRIES:
      t1 = t0 + \
          np.timedelta64(
              int(analysis_metadata["pre_buffer"]), "s",) / np.timedelta64(1, 's')

      t2 = t1 + np.timedelta64(
          int(
              analysis_metadata["pre_buffer"]
              + analysis_metadata["event_time_window"]
              + analysis_metadata["mid_buffer"]
              + analysis_metadata["forecast_time_window"]
              + analysis_metadata["post_buffer"]
          ), "s",) / np.timedelta64(1, 's')

      try:
        stream_dict = get_stream_multiple_stations(
            station_list=[station_name],
            t1=UTCDateTime(t1),
            t2=UTCDateTime(t2),
            network=network,
            client=client,
        )

        event_dir = saving_dir / f"data/{t1.strftime('%Y-%m-%d_%H-%M-%S')}"

        os.makedirs(event_dir, exist_ok=True)

        with open(event_dir / "stream_dict.pkl", "wb") as f:
          pickle.dump(stream_dict, f)

        amplitude, times = raw_stream_to_amplitude_and_times(stream_dict[station_name])
        amplitude, times = raw_stream_to_amplitude_and_times(stream_dict[station_name])
        np.save(event_dir / "amplitude.npy", amplitude)
        np.save(event_dir / "times.npy", times)

        minus_time_range = [
            analysis_metadata["pre_buffer"],
            analysis_metadata["pre_buffer"]
            + analysis_metadata["event_time_window"],
        ]

        plus_time_range = [
            analysis_metadata["pre_buffer"]
            + analysis_metadata["event_time_window"],
            analysis_metadata["pre_buffer"]
            + analysis_metadata["event_time_window"]
            + analysis_metadata["mid_buffer"],
        ]

        a_minus = amplitude[
            (times >= minus_time_range[0]) & (times <= minus_time_range[1])
        ]
        a_plus = amplitude[
            (times >= plus_time_range[0]) & (times <= plus_time_range[1])
        ]

        a_max_minus.append(
            np.max(a_minus)
        )  # note that the instrument response is removed, then we detrend, and hi
        a_max_plus.append(np.max(a_plus))

      except Exception as e:
        print(f"Error dowloading event {i}: {e}")

      i += 1
  np.save(saving_dir / "a_max_minus.npy", np.array(a_max_minus))
  np.save(saving_dir / "a_max_plus.npy", np.array(a_max_plus))


def main(_):
  start_times = _parse_start_times(_START_TIMES.value)
  # create the metadata
  metadata, analysis_metadata, saving_dir = create_and_save_metadata(
      start_times,
      # stname=_DEF_STATIONS,
      # network=_DEF_NETWORK,
      # org=_DEF_ORG,
      number_of_tries=_DEF_NUM_TRIES,
      mid_buffer=_MID_BUFFER.value,
      forecast_time_window=_FORECAST_TIME_WINDOW.value,
      event_time_window=_EVENT_TIME_WINDOW.value,
      shift_times=_SHIFT_TIMES.value,
  )

  download_waveforms(start_times, metadata, saving_dir)


if __name__ == '__main__':
  app.run(main)

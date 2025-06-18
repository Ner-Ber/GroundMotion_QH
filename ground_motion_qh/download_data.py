"""Download and preprocess waveforms for specified times.

Example usage, from terminal:
# activate python env for this project 

$ conda activate ground_motion_qh

# run the script from terminal. An example:
# (Only 'start_times' is required, all other flags are optional)

$ python3 ./ground_motion_qh/download_data.py \
  --start_times='(2023, 10, 1, 0, 0, 0), (2024, 3, 7, 15, 20, 55), ' \
  --data_dir='path/to/data/parent/dir' \
  --override_download_name=False \
  --name_prefix='ANZA' \
  --mid_buffer=20 \
  --forecast_time_window=14400 \
  --event_time_window=30 \
  --shift_times=True \
  --station='SND' \
  


"""

import datetime
from typing import Sequence
from absl import app
from absl import flags
import numpy as np
import obspy as obs
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
import ground_motion_qh
from ground_motion_qh import get_waveforms
import os
from pathlib import Path
import pickle
from obspy.taup import TauPyModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

model = TauPyModel(model="iasp91")

BASE_DIR = Path(ground_motion_qh.__file__).parent.parent

# DEFAULTS
_DEF_STATIONS = "SND"
_DEF_NETWORK = "AZ"
_DEF_ORG = "IRIS"
_DEF_NUM_TRIES = 2
_DEF_MID_BUFFER = 20
_DEF_FORECAST_TIME_WINDOW = 4*60*60
_DEF_EVENT_TIME_WINDOW = 30
_DEF_SHIFT_TIMES = True  # @keliankaz what is this for?


# Command line flags
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
  logging.info(f"Determined download name: {download_name}")
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

  logging.info(f"Target download directory: {parsed_dir}")
  if _OVERRIDE_DOWNLOAD_NAME.value is False and parsed_dir.exists():
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_parsed_dir = parsed_dir.parent / f"{parsed_dir.name}_{date_str}"
    logging.warning(
        f"Directory {parsed_dir} already exists and override is False. Creating new directory: {new_parsed_dir}")
    parsed_dir = new_parsed_dir
  elif _OVERRIDE_DOWNLOAD_NAME.value is True and parsed_dir.exists():
    logging.warning(
        f"Directory {parsed_dir} already exists and override is True. Contents may be overwritten.")

  os.makedirs(parsed_dir, exist_ok=True)
  logging.debug(f"Ensured download directory exists: {parsed_dir}")
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
  logging.info(f"Saving {len(start_times)} start times to {saving_path}")
  with open(saving_path, 'w') as f:
    for t1 in start_times:
      # format the UTCDateTime to a string
      t1_str = format_string_from_utcdatetime(t1)
      # write the time pair to the file
      f.write(f"{t1_str}\\n")
  logging.debug(f"Successfully saved start times to {saving_path}")


# Renamed start_times to start_times_input for clarity from outer scope
def _parse_start_times(start_times_input: str):
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
  logging.info(
      f"Parsing start times from: {'file' if os.path.isfile(start_times_input) else 'string'}")
  if not start_times_input:
    logging.error("Start times input is empty or None.")
    raise ValueError("Start times input cannot be empty.")
  if os.path.isfile(start_times_input):
    parsed_times = _parse_pairs_file(start_times_input)
    logging.info(
        f"Parsed {len(parsed_times)} start times from file: {start_times_input}")
    return parsed_times
  else:
    parsed_times = _parse_times_string(start_times_input)
    logging.info(f"Parsed {len(parsed_times)} start times from string.")
    return parsed_times


def _parse_pairs_file(file_path: str):
  logging.debug(f"Reading start times from file: {file_path}")
  with open(file_path, 'r') as f:
    lines = f.readlines()
  return [_parse_time_line_from_file(line.strip('\\n')) for line in lines]


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
  reformatted_str = time_pair_string.replace(
      ' ', '').replace('[', '(').replace(']', ')')
  reformatted_str = reformatted_str.strip('[]').strip('()')
  time_strings = reformatted_str.split('),(')
  time_strings = [t.strip(' (),') for t in time_strings]
  times_utc = [_time_str_parts_to_utc_datetime(t) for t in time_strings]
  logging.debug(f"Parsed {len(times_utc)} UTCDateTime objects from string.")
  return times_utc


def create_and_save_metadata(
    start_times: Sequence[UTCDateTime],
    number_of_tries=_DEF_NUM_TRIES,
    mid_buffer=_DEF_MID_BUFFER,
    forecast_time_window=_DEF_FORECAST_TIME_WINDOW,
    event_time_window=_DEF_EVENT_TIME_WINDOW,
    shift_times=_DEF_SHIFT_TIMES,
):
  logging.info("Creating and saving metadata.")
  station_metadata = dict(
      stname=_STNAME.value,
      network=_NETWORK.value,
      org=_ORG.value,
      number_of_tries=number_of_tries,
  )
  logging.info(f"Station metadata: {station_metadata}")

  # EXPLANATION OF BUFFERS:
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

  logging.info(f"Analysis metadata: {analysis_metadata}")
  # pprint(analysis_metadata, width=1) # Original pprint for console view
  metadata = dict(
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
  logging.info(f"Attempting to save metadata to: {metadata_file_path}")
  try:
    np.save(metadata_file_path, metadata)
    logging.info(f"Metadata successfully saved to {metadata_file_path}")
  except Exception as e:
    logging.error(f"Failed to save metadata to {metadata_file_path}: {e}")

  start_times_file = download_dir / "start_times.txt"
  start_times_to_file(
      start_times, start_times_file
  )

  return metadata, analysis_metadata, download_dir


def download_waveforms(
    start_times: Sequence[UTCDateTime],
    metadata: dict,
    saving_dir: Path,
):
  logging.info(
      f"Starting waveform download process for {len(start_times)} events. Saving to: {saving_dir}")
  analysis_metadata = metadata["analysis_metadata"]
  station_metadata = metadata["station_metadata"]
  station_name = station_metadata["stname"]
  network = station_metadata["network"]
  org = station_metadata["org"]
  logging.info(f"Using FDSN client for organization: {org}")
  client = FDSN_Client(org)

  minus_time_range = [
      analysis_metadata["pre_buffer"],
      analysis_metadata["pre_buffer"]
      + analysis_metadata["event_time_window"],
  ]
  plus_time_range = [
      analysis_metadata["pre_buffer"]
      + analysis_metadata["event_time_window"]
      + analysis_metadata["mid_buffer"],
      analysis_metadata["pre_buffer"]
      + analysis_metadata["event_time_window"]
      + analysis_metadata["mid_buffer"]
      + analysis_metadata["forecast_time_window"],
  ]

  time_paris = []
  for i_event, t1 in enumerate(start_times):
    # create pairs for t1 and t2
    t2 = t1 + np.timedelta64(
        int(
            analysis_metadata["pre_buffer"]
            + analysis_metadata["event_time_window"]
            + analysis_metadata["mid_buffer"]
            + analysis_metadata["forecast_time_window"]
            + analysis_metadata["post_buffer"]
        ), "s",) / np.timedelta64(1, 's')
    time_paris.append((UTCDateTime(t1), UTCDateTime(t2)))
  waveforms_bulk = get_waveforms.get_bulk_streams_single_station(
      station=station_name,
      time_pairs=time_paris,
      network=network,
      client=client,
  )

  a_max_minus = []
  a_max_plus = []
  checked_starttimes = set()
  for i_event, wave_form in enumerate(waveforms_bulk):
    starttime = wave_form.meta.starttime
    if starttime.timestamp in checked_starttimes:
      continue
    checked_starttimes.add(starttime.timestamp)
    all_components_st = obs.Stream(
        [wf for wf in waveforms_bulk if wf.meta.starttime == starttime])

    event_dir = saving_dir / \
        f"data/{starttime.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(event_dir, exist_ok=True)
    logging.info(f"Event data will be saved to: {event_dir}")

    # with open(event_dir / "stream.pkl", "wb") as f:
    # pickle.dump(stream_dict, f)
    # logging.debug(f"Saved stream_dict.pkl for event {i_event + 1}")
    all_components_st.write(event_dir / "stream.mseed", format="MSEED")
    logging.debug(f"Saved stream.mseed for event {i_event + 1}")

    amplitude, times = get_waveforms.raw_stream_to_amplitude_and_times(
        all_components_st
    )
    np.save(event_dir / "amplitude.npy", amplitude)
    np.save(event_dir / "times.npy", times)
    logging.debug(f"Saved amplitude.npy and times.npy for event {i_event + 1}")

    a_minus = amplitude[
        (times >= minus_time_range[0]) & (times <= minus_time_range[1])
    ]
    a_plus = amplitude[
        (times >= plus_time_range[0]) & (
            times <= plus_time_range[1])
    ]

    a_max_minus.append(
        np.max(a_minus)
    )
    a_max_plus.append(np.max(a_plus))
    logging.debug(f"Calculated and appended a_max values for event {i_event + 1}")

  np.save(saving_dir / "a_max_minus.npy", np.array(a_max_minus))
  np.save(saving_dir / "a_max_plus.npy", np.array(a_max_plus))
  logging.info(f"Saved a_max_minus.npy and a_max_plus.npy to {saving_dir}")


def main(_):
  logging.info("Starting main execution.")
  if not _START_TIMES.value:
    logging.error("CRITICAL: --start_times flag is required but not provided.")
    print("Error: The --start_times flag must be provided. Use --help for more information.")
    return

  try:
    start_times = _parse_start_times(_START_TIMES.value)
    if not start_times:
      logging.warning("No start times were parsed. Exiting.")
      return
  except ValueError as e:
    logging.error(f"Failed to parse start times: {e}", exc_info=True)
    return

  try:
    metadata, _, saving_dir = create_and_save_metadata(
        start_times,
        number_of_tries=_DEF_NUM_TRIES,
        mid_buffer=_MID_BUFFER.value,
        forecast_time_window=_FORECAST_TIME_WINDOW.value,
        event_time_window=_EVENT_TIME_WINDOW.value,
        shift_times=_SHIFT_TIMES.value,
    )
  except Exception as e:
    logging.error(f"Failed to create and save metadata: {e}", exc_info=True)
    return

  try:
    download_waveforms(start_times, metadata, saving_dir)
    logging.info("Waveform download process completed.")
  except Exception as e:
    logging.error(f"Waveform download process failed: {e}", exc_info=True)

  logging.info("Main execution finished.")


if __name__ == '__main__':
  app.run(main)

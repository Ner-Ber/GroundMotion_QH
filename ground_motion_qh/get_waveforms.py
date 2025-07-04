import logging
from typing import Sequence
import numpy as np
from obspy import UTCDateTime
import obspy as obs
from obspy.clients.fdsn import Client as FDSN_Client

"""
Example of usage w/ defaults:
stream_dict = get_waveforms.get_stream_multiple_stations()
amplitude, times = get_waveforms.raw_stream_to_amplitude_and_times(stream_dict['SND'])

"""
T1 = UTCDateTime("2009-03-24")
T2 = UTCDateTime("2009-03-30")
STN_LIST = ['SND']
NETWORK = 'AZ'
ORG = 'IRIS'
CLIENT = FDSN_Client(ORG)
PRE_FILT = [0.001, 0.005, 45, 50]


def get_bulk_streams_single_station(
    station: str = STN_LIST[0],
    time_pairs: Sequence[tuple[UTCDateTime, UTCDateTime]] = [(T1, T2)],
    network: str = NETWORK,
    client=CLIENT
) -> obs.Stream:
  """
  Get waveforms for a single station across multiple time ranges.

  Args:
    station: Station code
    time_pairs: Sequence of (start_time, end_time) tuples
    network: Network code
    client: FDSN client

  Returns:
    obspy.Stream object containing the requested waveforms
  """
  bulk = []
  for t1, t2 in time_pairs:
    bulk.append((network, station, "*", "HH*", t1, t2))

  try:
    bulk_waveforms = client.get_waveforms_bulk(bulk, attach_response=True)
    bulk_waveforms = bulk_waveforms.sort(['starttime', 'endtime', 'channel'])
  except Exception as e:
    logging.error("Error fetching waveforms: %s", e)
    bulk_waveforms = obs.Stream()
  return bulk_waveforms


def get_stream_multiple_stations(
    station_list: list[str] = STN_LIST,
    t1: UTCDateTime = T1,
    t2: UTCDateTime = T2,
    network: str = NETWORK,
    client=CLIENT
) -> dict[str, obs.Stream]:
  """
  Get waveforms for multiple stations in a given time range.
  """
  station_streams = {}
  for stn in station_list:
    station_streams[stn] = client.get_waveforms(network, str(stn), "*", "HH*", t1, t2,attach_response=True)
  return station_streams


def preprocess_waveforms(
    stn_waveform_strm: obs.Stream,
    detrend: bool = True,
    taper: bool = True,
    remove_response: bool = True,
    filter: bool = True,
    filter_type: str = 'highpass',
    freq: float = 0.01,
    output: str = "VEL",  # can be "VEL" or "ACC"
    client=CLIENT,
) -> obs.Stream:
  if detrend:
    stn_waveform_strm.detrend()
  if taper:
    stn_waveform_strm.taper(max_percentage=0.05)
  if filter:
    stn_waveform_strm.filter(filter_type, freq=freq)
  if remove_response:
    try:
      stn_waveform_strm = stn_waveform_strm.remove_response(
          inventory=None, water_level=None, pre_filt=PRE_FILT, output=output)
    except Exception as e:
      logging.error(
          "Error removing response: %s \\n will connect to client to download response", e)
      # Attempt to fetch inventory if not provided
      inv = client.get_stations(
          station=stn_waveform_strm[0].meta.station, level="response",
          starttime=stn_waveform_strm[0].meta.starttime, endtime=stn_waveform_strm[0].meta.endtime)
      stn_waveform_strm = stn_waveform_strm.remove_response(
          inventory=inv, water_level=None, pre_filt=PRE_FILT, output=output)
  if taper:
    stn_waveform_strm.taper(max_percentage=0.05)
  if filter:
    stn_waveform_strm.filter(filter_type, freq=freq)
  return stn_waveform_strm


def combine_components(
    stn_waveform_strm: obs.Stream,
) -> np.ndarray:
  channels_combined = []
  squared_data = np.zeros_like(stn_waveform_strm[0].data)
  for trace in stn_waveform_strm:
    channels_combined.append(trace.meta.channel)
    squared_data = squared_data + trace.data**2
  trace_amplitude = np.sqrt(squared_data)
  logging.info("Combining components: %s", channels_combined)
  return trace_amplitude


def raw_stream_to_amplitude_and_times(
    raw_stn_waveform_strm: obs.Stream,
) -> tuple[np.ndarray, np.ndarray]:
  stn_waveform_strm_preprcsd = preprocess_waveforms(raw_stn_waveform_strm)
  stn_waveform_strm_amp = combine_components(stn_waveform_strm_preprcsd)
  time_vector = raw_stn_waveform_strm[0].times()
  return stn_waveform_strm_amp, time_vector

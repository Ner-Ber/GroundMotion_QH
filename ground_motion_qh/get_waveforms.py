import logging
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


def get_stream_multiple_stations(
    station_list: list[str] = STN_LIST,
    t1: UTCDateTime = T1,
    t2: UTCDateTime = T2,
    network: str = NETWORK,
    client=CLIENT
) -> dict[str, obs.Stream]:
  station_streams = {}
  for stn in station_list:
    station_streams[stn] = client.get_waveforms(network, str(stn), "*", 'HH*', t1, t2)
    # stn_waveform = obs.Stream()
    # for i, comp in enumerate(['HHE', 'HHN', 'HHZ']):
    #   stn_waveform += client.get_waveforms(network, str(stn), "*", str(comp), t1, t2)
    # station_streams[stn] = stn_waveform
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
  if remove_response:
    inv = client.get_stations(
        station=stn_waveform_strm[0].meta.station, level="response",
        starttime=stn_waveform_strm[0].meta.starttime, endtime=stn_waveform_strm[0].meta.endtime)
    stn_waveform_strm = stn_waveform_strm.remove_response(inventory=inv, water_level=None,
                                                          pre_filt=PRE_FILT, output=output)
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
    raw_stn_waveform_strm
) -> tuple[np.ndarray, np.ndarray]:
  stn_waveform_strm_preprcsd = preprocess_waveforms(raw_stn_waveform_strm)
  stn_waveform_strm_amp = combine_components(stn_waveform_strm_preprcsd)
  time_vector = raw_stn_waveform_strm[0].times()
  return stn_waveform_strm_amp, time_vector

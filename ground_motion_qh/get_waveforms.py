from obspy import UTCDateTime
import obspy as obs
from obspy.clients.fdsn import Client as FDSN_Client


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
    stn_waveform = obs.Stream()
    for i, comp in enumerate(['HHE', 'HHN', 'HHZ']):
      stn_waveform += client.get_waveforms(network, str(stn), "*", str(comp), t1, t2)
    station_streams[stn] = stn_waveform
  return station_streams


def preprocess_waveforms(
    stn_waveform: obs.Stream,
    detrend: bool = True,
    taper: bool = True,
    filter: bool = True,
    filter_type: str = 'highpass',
    freq: float = 0.01,
    client=CLIENT,
    station_name: str = STN_LIST[0],
) -> obs.Stream:
  inv = client.get_stations(
      station=station_name, level="response", starttime=t, endtime=(t+3800))
  stn_waveform = stn_waveform.remove_response(inventory=inv, water_level=None,
                                              pre_filt=PRE_FILT, output="VEL")
  if detrend:
    stn_waveform.detrend()
  if taper:
    stn_waveform.taper(max_percentage=0.05)
  if filter:
    stn_waveform.filter(filter_type, freq=freq)
  return stn_waveform

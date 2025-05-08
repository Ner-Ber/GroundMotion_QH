from obspy import read, read_inventory,UTCDateTime
import numpy as np
import matplotlib.pyplot as plt

#1.cut wave by time
def cut_wave(stream,starttime,endtime):
    stream.merge(method=1, fill_value=None)
    cutst = st.slice(starttime,endtime)
    return cutst

#2.remove response
def remove_response(stream,inv):
    pre_filt = [0.01, 0.02, 45, 50]
    output = "VEL"
    stream.remove_response(inventory=inv, pre_filt=pre_filt, output=output, water_level=60)
    return stream

#3.calculate PGV
def calculate_PGV(trace):
    PGV = max(abs(trace.data))
    PGV_index = np.argmax(trace.data)
    print(f"PGV: {PGV:.6f} m/s")
    return PGV,PGV_index

# 4.plot waveform with PGV marked
def plot_waveform(trace, pgv, pgv_index):
    print(trace)
    sampling_rate = trace.stats.sampling_rate
    time = np.arange(len(trace.data)) / sampling_rate
    pgv_time = time[pgv_index]
    print(f"PGV time point: {pgv_time}s (index:{pgv_index})")

    start_Pre_Buffer = max(pgv_time - 230, time[0])
    start_Tw_minus = max(pgv_time - 10, time[0])
    start_Mid_Buffer = min(pgv_time + 20, time[-1])
    start_Tw_plus = min(pgv_time + 30, time[-1])
    start_Post_Buffer = min(pgv_time + 3900, time[-1])
    end_Post_Buffer = min(pgv_time + 4120, time[-1])

    # find the index of  start_Tw_minus & start_Mid_Buffer
    idx_start = np.searchsorted(time, start_Tw_minus)
    idx_end = np.searchsorted(time, start_Mid_Buffer)

    # get the Tw_minus data
    cut_time_window = time[idx_start:idx_end]
    cut_data_window = trace.data[idx_start:idx_end]

    plt.figure(figsize=(15, 12))
    plt.subplot(2, 1, 1)
    plt.plot(time, trace.data, 'b-', label="Velocity Waveform",linewidth=1)

    # different colors for every duration
    infos = {
        'Pre_Buffer': ('gray', 'Pre_Buffer/220s'),
        'Tw_Minus': ('yellow', 'Tw_Minus/30s'),
        'Mid_Buffer': ('gray', 'Mid_Buffer/10s'),
        'Tw_Plus': ('cyan', 'Tw_Plus/3600s'),
        'Post_Buffer': ('gray', 'Post_Buffer/10s')
    }

    plt.axvspan(start_Pre_Buffer, start_Tw_minus,
                color=infos['Pre_Buffer'][0], alpha=0.2,
                label=f"{infos['Pre_Buffer'][1]}")
    plt.axvspan(start_Tw_minus, start_Mid_Buffer,
                color=infos['Tw_Minus'][0], alpha=0.2,
                label=f"{infos['Tw_Minus'][1]}")
    plt.axvspan(start_Mid_Buffer, start_Tw_plus,
                color=infos['Mid_Buffer'][0], alpha=0.2,
                label=f"{infos['Mid_Buffer'][1]}")
    plt.axvspan(start_Tw_plus, start_Post_Buffer,
                color=infos['Tw_Plus'][0], alpha=0.2,
                label=f"{infos['Tw_Plus'][1]}")
    plt.axvspan(start_Post_Buffer, end_Post_Buffer,
                color=infos['Post_Buffer'][0], alpha=0.2,
                label=f"{infos['Post_Buffer'][1]}")

    # PGV
    plt.scatter(pgv_time, trace.data[pgv_index], color='red', s=80,label=f'PGV: {pgv:.3e} m/s',zorder=5)
    plt.axvline(x=pgv_time, color='r', linestyle=':', alpha=0.7)
    plt.annotate(f"PGV\n{pgv:.3e} m/s",
                 xy=(pgv_time, trace.data[pgv_index]),
                 xytext=(10, 20), textcoords='offset points',
                 ha='left', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                 arrowprops=dict(arrowstyle="->", color='red'))

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    title = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.channel}"
    plt.title(f"Waveform Analysis\n{title}")
    plt.xlim([time[0], time[-1]])

    plt.legend(bbox_to_anchor=(0.5, -0.1),
               loc='upper center', ncol=4,
               fontsize=14, framealpha=0.8)

    plt.grid(True, linestyle=':', alpha=0.5)
    plt.subplots_adjust(bottom=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(cut_time_window,cut_data_window, 'b-', label="Velocity Waveform (Tw_minus)", linewidth=1)
    plt.axvspan(start_Tw_minus, start_Mid_Buffer, color='yellow', alpha=0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title(f"Waveform Analysis(Tw_minus)\n{title}")
    plt.xlim([cut_time_window[0], cut_time_window[-1]])

    plt.scatter(pgv_time, trace.data[pgv_index], color='red', s=80, label=f'PGV: {pgv:.3e} m/s', zorder=5)
    plt.axvline(x=pgv_time, color='r', linestyle=':', alpha=0.7)

    plt.annotate(f"PGV\n{pgv:.3e} m/s",
                 xy=(pgv_time, trace.data[pgv_index]),
                 xytext=(10, 20), textcoords='offset points',
                 ha='left', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                 arrowprops=dict(arrowstyle="->", color='red'))

    plt.legend(bbox_to_anchor=(0.5, 0.15),
               loc='upper center', ncol=6,
               fontsize=14, framealpha=0.8)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()

if __name__ == "__main__":
    st = read("AZ.SND.HHE.20151003T0000.ms")
    inv = read_inventory("AZ.SND_response.xml", "STATIONXML")
    starttime = UTCDateTime("2015-10-03T20:50:00")
    endtime = UTCDateTime("2015-10-03T22:20:00")

    #1.propress wave data
    st = cut_wave(st, starttime, endtime)
    st.detrend("linear").detrend("demean")
    # 2.remove response
    st = remove_response(st, inv)

    #3.PGV
    tr = st[0]
    PGV,PGV_index = calculate_PGV(tr)
    # 4.plot PGV
    plot_waveform(tr, PGV, PGV_index)
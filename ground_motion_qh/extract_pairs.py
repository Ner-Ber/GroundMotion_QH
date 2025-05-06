import numpy as np

TW_MINUS = 10
TW_BUFFER = 10
TW_PLUS = 100


def create_tw_minus_slices(
    tau_vec,
    time_vector,  # should be sorted
    tw_minus,
):
    tw_minus_ends = np.searchsorted(time_vector, tau_vec, side='left') + 1
    tw_minus_starts = np.searchsorted(
        time_vector, tau_vec - tw_minus, side='right')
    slices_list = [slice(tw_minus_starts[i], tw_minus_ends[i])
                   for i in range(len(tw_minus_ends))]
    return slices_list


def create_tw_buffer_slices(
    tau_vec,
    time_vector,  # should be sorted
    tw_buffer,
):
    tw_buffer_starts = np.searchsorted(time_vector, tau_vec, side='left')
    tw_buffer_ends = np.searchsorted(
        time_vector, tau_vec + tw_buffer, side='left')
    slices_list = [slice(tw_buffer_starts[i], tw_buffer_ends[i])
                   for i in range(len(tw_buffer_ends))]
    return slices_list


def create_tw_plus_slices(
    tau_vec,
    time_vector,  # should be sorted
    tw_plus,
    tw_buffer,
):
    tw_plus_starts = np.searchsorted(
        time_vector, tau_vec + tw_buffer, side='left')
    tw_plus_ends = np.searchsorted(
        time_vector, tau_vec + tw_plus + tw_buffer, side='left')
    slices_list = [slice(tw_plus_starts[i], tw_plus_ends[i])
                   for i in range(len(tw_plus_ends))]
    return slices_list


def slices_to_max_amp_pairs(
    waveform,
    tw_minus_slices,
    tw_plus_slices,
):
    a_minus_vector = []
    a_plus_vector = []
    for i in range(len(tw_minus_slices)):
        try:
            max_of_minus = np.max(np.abs(waveform[tw_minus_slices[i]] -
                                  np.mean(waveform[tw_minus_slices[i]])))
        except ValueError:
            max_of_minus = np.nan
        try:
            max_of_plus = np.max(np.abs(waveform[tw_plus_slices[i]] -
                                 np.mean(waveform[tw_plus_slices[i]])))
        except ValueError:
            max_of_plus = np.nan
        a_minus_vector.append(max_of_minus)
        a_plus_vector.append(max_of_plus)
    return np.array(a_minus_vector), np.array(a_plus_vector)


def waveform_to_max_amp_pairs(
    tau_vec,
    waveform,
    time_vector,
    tw_minus=TW_MINUS,
    tw_plus=TW_PLUS,
    tw_buffer=TW_BUFFER,
):
    tw_minus_slices = create_tw_minus_slices(
        tau_vec,
        time_vector,
        tw_minus=tw_minus,
    )
    tw_plus_slices = create_tw_plus_slices(
        tau_vec,
        time_vector,
        tw_plus=tw_plus,
        tw_buffer=tw_buffer,
    )
    a_minus_vector, a_plus_vector = slices_to_max_amp_pairs(
        waveform,
        tw_minus_slices,
        tw_plus_slices,
    )
    return a_minus_vector, a_plus_vector

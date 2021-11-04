import mne
import os
import scipy.io
import numpy as np
from mne import EpochsArray
import pandas as pd


def convert_to_mne(path):
    sfreq = 250
    highpass = 2
    lowpass = 30
    ch_names = ['C3', 'Cz', 'C4', 'CPz', 'P3', 'Pz', 'P4', 'POz']
    device = "g.Nautilus system (g.tec medical engineering GmbH, Austria)"
    tmin = -0.200
    tmax = 1.200
    n_stimuli = 8
    train_runs_per_block = 10

    dirname = os.path.dirname(path)
    filename = os.path.basename(path)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    info.set_montage('standard_1020')
    info['highpass'] = highpass
    info['lowpass'] = lowpass
    info['device_info'] = dict(type=device)

    # Load data and labels
    epochs_data = scipy.io.loadmat(path)[os.path.splitext(filename)[0]]
    epochs_data = epochs_data.transpose((2, 0, 1))
    epochs_data /= 1e6

    stimuli_filename = filename.replace('Data.mat', 'Events.txt')
    stimuli_path = os.path.join(dirname, stimuli_filename)
    with open(stimuli_path) as stimuli_file:
        stimuli = np.array(stimuli_file.read().splitlines())

    cues_filename = filename.replace('Data.mat', 'Labels.txt')
    cues_path = os.path.join(dirname, cues_filename)
    with open(cues_path) as labels_file:
        cues = np.array(labels_file.read().splitlines())

    if filename.startswith('test'):
        runs_per_block_path = os.path.join(filename,
                                           f'runs_per_block.txt')
        with open(runs_per_block_path) as runs_per_block_file:
            runs_per_block = int(runs_per_block_file.read())
    else:
        runs_per_block = train_runs_per_block

    # Create events
    samples_per_epoch = (tmax - tmin) * info['sfreq']
    n_epochs = len(stimuli)
    events = np.zeros((n_epochs, 3), dtype=int)
    events[:, 2] = stimuli
    events[:, 0] = np.arange(
        start=-tmin * info['sfreq'], step=samples_per_epoch,
        stop=n_epochs * samples_per_epoch)
    event_id = {f'stimulus/{i + 1}': i + 1 for i in range(n_stimuli)}

    # Create metadata
    epochs_per_run = n_stimuli
    epochs_per_block = epochs_per_run * runs_per_block
    n_blocks = len(cues)
    cues = np.repeat(cues, epochs_per_block)
    blocks = np.repeat(np.arange(n_blocks), epochs_per_block)
    runs = np.tile(np.repeat(np.arange(runs_per_block), epochs_per_run),
                   n_blocks)

    metadata = pd.DataFrame({
        'cue': cues,
        'stimulus': stimuli,
        'block': blocks,
        'run': runs,
    }).convert_dtypes()

    # Create epochs
    return EpochsArray(events=events, event_id=event_id,
                       metadata=metadata, data=epochs_data,
                       info=info, tmin=tmin)

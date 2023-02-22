import mne
import pandas
import dn3
from dn3.configuratron import ExperimentConfig
import os
from tqdm import tqdm
import numpy as np
import torch

mapping = {'EEG FP1-REF': 'FP1', 'EEG FP2-REF': 'FP2', 'EEG F3-REF': 'F3', 'EEG F4-REF': 'F4', 'EEG C3-REF': 'C3', 'EEG C4-REF': 'C4', 'EEG P3-REF': 'P3', 'EEG P4-REF': 'P4', 'EEG O1-REF': 'O1', 'EEG O2-REF': 'O2', 'EEG F7-REF': 'F7', 'EEG F8-REF': 'F8', 'EEG T3-REF': 'T3', 'EEG T4-REF': 'T4', 'EEG T5-REF': 'T5', 'EEG T6-REF': 'T6', 'EEG A1-REF': 'A1', 'EEG A2-REF': 'A2', 'EEG FZ-REF': 'FZ', 'EEG CZ-REF': 'CZ', 'EEG PZ-REF': 'PZ'}

names = [name[:-4] for name in os.listdir('data/datasets/tuh_eeg_artifact/v3.0.0/edf/01_tcp_ar')]
names = list(set(names))
names = [name for name in names if not name.endswith("_seiz")]

pbar = tqdm(total=len(names))

for index, name in enumerate(names):
    raw = mne.io.read_raw_edf('data/datasets/tuh_eeg_artifact/v3.0.0/edf/01_tcp_ar/{}.edf'.format(name), preload=True)
    drop = [name for name in raw.ch_names if name not in mapping.values()]

    mne.datasets.eegbci.standardize(raw)  # Set channel names
    raw = raw.set_eeg_reference(ref_channels='average')
    montage = mne.channels.make_standard_montage('standard_1020')
    raw = raw.rename_channels(mapping)
    raw = raw.drop_channels(drop, on_missing = 'ignore')
    raw = raw.resample(256)
    raw = raw.filter(0.1, 80)
    raw = raw.anonymize()

    assert len(raw.ch_names) == 21

    csv = pandas.read_csv("data/datasets/tuh_eeg_artifact/v3.0.0/edf/01_tcp_ar/{}.csv".format(name), skiprows=6)
    csv = csv.drop(columns=["channel", "confidence"])
    csv = csv.drop_duplicates(subset=('start_time', 'stop_time', 'label'))
    csv = csv.sort_values(['start_time', 'stop_time', 'label'])
    csv = csv.reset_index(drop=True)

    booleans = np.zeros(len(csv), dtype=bool)
    booleans[0] = True
    idx = 0

    for i in range(1, len(csv)):
        if booleans[idx]:
            if csv.iloc[idx].stop_time < csv.iloc[i].start_time:
                idx = i
                booleans[i] = True

    csv = csv[booleans].reset_index(drop=True)

    onset = csv.start_time.values
    duration = csv.stop_time.values - csv.start_time.values
    description = csv.label.values

    annotations = mne.Annotations(onset, duration, description)
    raw = raw.set_annotations(annotations)

    dirname = '/home/s194260/BENDR/data/datasets/artifact/' + name.split('_')[1]
    filename = '/home/s194260/BENDR/data/datasets/artifact/' + name.split('_')[1] + '/' + name + '.edf'

    if not os.path.exists(dirname): os.makedirs(dirname)

    mne.export.export_raw(filename, raw, overwrite=True)

    pbar.set_description(name)
    pbar.update(1)


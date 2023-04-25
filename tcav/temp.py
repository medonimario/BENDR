import os, random, shutil
from tqdm import tqdm
import mne
import os
import numpy as np
from tqdm import tqdm

def rename(name):
    name = name.rstrip('.').upper()
    name = name.replace("EEG ", "")
    name = name.replace("-REF", "")
    return name

def pick_and_rename_channels(raw):
    if 'P7' in raw.ch_names:
        raw.rename_channels({'P7': 'T5'})
    if 'P8' in raw.ch_names:
        raw.rename_channels({'P8': 'T6'})
    if 'T3' in raw.ch_names:
        raw.rename_channels({'T3': 'T7'})
    if 'T4' in raw.ch_names:
        raw.rename_channels({'T4': 'T8'})

    EEG_20_div = [
                'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'T5', 'P3', 'PZ', 'P4', 'T6',
                 'O1', 'O2'
    ]
    
    raw.pick_channels(ch_names=EEG_20_div)
    raw.reorder_channels(EEG_20_div)

    return raw

dir_path = "/nobackup/tsal-tmp/tuh_eeg"

with tqdm() as pbar:
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".edf"):
                if random.random() < 0.01:
                    try:
                        raw = mne.io.read_raw_edf(os.path.join(root, file), verbose=False, preload=True)
                        raw = raw.copy()
                        mne.datasets.eegbci.standardize(raw)  # Set channel names
                        new_names = dict((ch_name, rename(ch_name)) for ch_name in raw.ch_names)
                        raw.rename_channels(new_names)   
                        raw = raw.set_eeg_reference(ref_channels='average', projection=True, verbose = False)
                        raw = raw.apply_proj(verbose=False)

                        montage = mne.channels.make_standard_montage('standard_1020')
                        raw = raw.set_montage(montage, on_missing='ignore'); # Set montage
                        raw = pick_and_rename_channels(raw)
                        raw = raw.resample(256)
                    
                        # Save the raw data in /scratch/s194260/tuh_eeg_preprocessed as edf file
                        raw.export(os.path.join('/scratch/s194260/tuh_eeg_preprocessed', file), verbose=False, overwrite=True)
                        pbar.update(1)           
                    
                    except Exception as e:
                        continue
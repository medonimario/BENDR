import os
if not os.getcwd().endswith('BENDR'): os.chdir(os.path.dirname(os.getcwd()))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from tqdm import tqdm

from importlib import reload
import BENDR.BENDR_utils as BENDR_utils
reload(BENDR_utils)
from BENDR.BENDR_utils import LinearBENDR, LinearHeadBENDR
from matplotlib import pyplot as plt
import activation_generator as act_gen
import numpy as np
import tcav as tcav
import tensorflow as tf

from pathlib import Path
folder_path = Path('/mnt/c/Users/anders/OneDriveDTU/Dokumenter/BENDR')

encoder_weights = folder_path / 'encoder_BENDR_linear_2_1024_20.pt'
enc_augment_weights = folder_path / 'enc_augment_BENDR_linear_2_1024_20.pt'
classifier_weights = folder_path / 'classifier_BENDR_linear_2_1024_20.pt'
extended_classifier_weights = folder_path / 'extended_classifier_BENDR_linear_2_1024_20.pt'

model = LinearBENDR(targets=2, samples=1024, channels=20, device='cuda')
model.load_all(encoder_weights, enc_augment_weights, classifier_weights, extended_classifier_weights)
model = model.train(False)
#model = model.to(torch.device('cpu'))

source_dir = 'data/concepts'
results_dir =  'data/tcav_results' 
activation_dir = 'data/activations'
cav_dir = None #'data/cavs'
bottlenecks = ['summarizer'] #['extended_classifier', 'summarizer', 'classifier']
alphas = [0.1]

target =  'left' #'Right fist, performed'

# concepts are stored in folders with these names
concepts =  ['fake', 'motor_lh', 'motor_rh'] #['Dorsal Stream Visual Cortex-lh', 'DorsoLateral Prefrontal Cortex-rh'] #['eyem', 'musc'] #['spsw', 'artf', 'bckg', 'gped', 'pled', 'eyem', 'musc']

labels = ['left', 'right']

from model import EEGWrapper, BENDR_cutted
import activation_generator as act_gen

class BENDRWrapper(EEGWrapper) : 
    def __init__(self, model, labels, sample_length_target):
        eeg_shape = [1, 20, sample_length_target]
        super(BENDRWrapper, self).__init__(eeg_shape=eeg_shape, eeg_labels=labels)
        self.model = model
        self.model_name = 'BENDR'

    def forward(self, x):
        return self.model.forward()

    def get_cutted_model(self, bottleneck):
        return BENDR_cutted(self.model, bottleneck)
    
tcav_model = BENDRWrapper(model, labels, 1024)

act_generator = act_gen.EEGActivationGenerator(
   tcav_model, source_dir, activation_dir, max_examples=25
   )

tf.compat.v1.logging.set_verbosity(2)
num_random_exp = 20

my_tcav = tcav.TCAV(target,
                   concepts,
                   bottlenecks,
                   act_generator,
                   alphas,
                   cav_dir=cav_dir,
                   num_random_exp=num_random_exp)

print('Loading mytcav')
results = my_tcav.run(run_parallel = False)

# Save dictionary that also contains numpy array
import pickle
with open('tcav_results_summarizer.pkl', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

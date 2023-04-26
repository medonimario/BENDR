import os, sys
if not os.getcwd().endswith('BENDR'): os.chdir(os.path.dirname(os.getcwd()))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from tqdm import tqdm
sys.path.append("/home/s194260/BENDR")

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
folder_path = Path('/home/s194260/BENDR')

encoder_weights = folder_path / 'encoder_BENDR_linear_2_1024_20.pt'
enc_augment_weights = folder_path / 'enc_augment_BENDR_linear_2_1024_20.pt'
classifier_weights = folder_path / 'classifier_BENDR_linear_2_1024_20.pt'
extended_classifier_weights = folder_path / 'extended_classifier_BENDR_linear_2_1024_20.pt'

model = LinearBENDR(targets=2, samples=1024, channels=20, device='cuda')
model.load_all(encoder_weights, enc_augment_weights, classifier_weights, extended_classifier_weights)
model = model.train(False)
#model = model.to(torch.device('cpu'))

source_dir = '/scratch/s194260/concepts_tuh' #'/scratch/s194260/concepts_mmidb_tasks_T0' #'/scratch/s194260/concepts_tuh'
results_dir =  '/scratch/s194260/tcav_results' 
activation_dir = '/scratch/s194260/activations2'
cav_dir = '/scratch/s194260/cavs2'
bottlenecks = ['encoder', 'enc_augment', 'summarizer', 'extended_classifier', 'classifier']
alphas = [0.1]

target =  'Left fist, performed'

# concepts are stored in folders with these names
#concepts = []
# concepts = ["Alpha_Dorsal Stream Visual Cortex-lh", "Alpha_Dorsal Stream Visual Cortex-rh",
#             "Alpha_Early Visual Cortex-lh", "Alpha_Early Visual Cortex-rh",
#             "Alpha_MT+ Complex and Neighboring Visual Areas-lh", "Alpha_MT+ Complex and Neighboring Visual Areas-rh",
#             "Alpha_Premotor Cortex-lh", "Alpha_Premotor Cortex-rh",
#             "Alpha_Primary Visual Cortex (V1)-lh","Alpha_Primary Visual Cortex (V1)-rh",
#             "Alpha_Somatosensory and Motor Cortex-lh", "Alpha_Somatosensory and Motor Cortex-rh",
#             "Alpha_Ventral Stream Visual Cortex-lh", "Alpha_Ventral Stream Visual Cortex-rh"]
#concepts = ['Left fist, performed 2', 'Right fist, performed 2']
concepts = ['random500_48', 'random500_49']

# concepts = ['Alpha_Somatosensory and Motor Cortex-lh', 'Alpha_Somatosensory and Motor Cortex-rh',
#             'Alpha_Primary Visual Cortex (V1)-lh', 'Alpha_Primary Visual Cortex (V1)-rh',
#             'Alpha_Orbital and Polar Frontal Cortex-lh', 'Alpha_Orbital and Polar Frontal Cortex-rh',
#             'Alpha_Early Visual Cortex-lh', 'Alpha_Early Visual Cortex-rh']

# concepts = ['Alpha_Premotor Cortex-lh', 'Alpha_Premotor Cortex-rh',
#             'Alpha_Early Visual Cortex-lh', 'Alpha_Early Visual Cortex-rh',
#             'Alpha_Orbital and Polar Frontal Cortex-lh', 'Alpha_Orbital and Polar Frontal Cortex-rh',
#             'Alpha_MT+ Complex and Neighboring Visual Areas-lh', 'Alpha_MT+ Complex and Neighboring Visual Areas-rh',
#             'Alpha_Dorsal Stream Visual Cortex-lh', 'Alpha_Dorsal Stream Visual Cortex-rh']

# Go through each folder in source_dir add add the name of the folder to concepts list if it has more than 25 .pkl files in it
# for concept in os.listdir(source_dir):
#     if len(os.listdir(os.path.join(source_dir, concept))) > 25:
#         concepts.append(concept)

labels = ['Left fist, performed', 'Right fist, performed']

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
num_random_exp = 25

my_tcav = tcav.TCAV(target,
                   concepts,
                   bottlenecks,
                   act_generator,
                   alphas,
                   cav_dir=cav_dir,
                   num_random_exp=num_random_exp)

print('Loading mytcav')
results = my_tcav.run(run_parallel = True)

# Save dictionary that also contains numpy array
import pickle
with open('tcav_results_left_random.pkl', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

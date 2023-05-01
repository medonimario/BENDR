import os, sys
sys.path.append(os.getcwd())

from BENDR.BENDR_utils import LinearBENDR
from matplotlib import pyplot as plt
import activation_generator as act_gen
import numpy as np
import tcav as tcav
import tensorflow as tf
from pathlib import Path
import datetime
from model import EEGWrapper, BENDR_cutted
import pickle
import activation_generator as act_gen
import argparse

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--concept_folder', default='class_data', type=str, help='Folder with concepts')
    parser.add_argument('--concepts', default=[], nargs='+', type=str, help='Concepts to run')
    parser.add_argument('--data_path', default='/work1/s194260/', type=str, help='Path to data')
    args = parser.parse_args()
    
    concept_folder = Path(args.concept_folder)
    concepts = args.concepts    
    data_path = Path(args.data_path)
    
    model_path = data_path / Path('checkpoints')

    encoder_weights = model_path / 'encoder_BENDR_linear_2_1024_20.pt'
    enc_augment_weights = model_path / 'enc_augment_BENDR_linear_2_1024_20.pt'
    classifier_weights = model_path / 'classifier_BENDR_linear_2_1024_20.pt'
    extended_classifier_weights = model_path / 'extended_classifier_BENDR_linear_2_1024_20.pt'

    model = LinearBENDR(targets=2, samples=1024, channels=20, device='cpu')
    model.load_all(encoder_weights, enc_augment_weights, classifier_weights, extended_classifier_weights)
    model = model.train(False)

    now = datetime.datetime.now()
    date = now.strftime("%m%d%H%M%S")

    source_dir = data_path / concept_folder
    results_dir =  data_path / 'tcav_results' 
    activation_dir = data_path / 'activations' / f'activations_{date}'
    cav_dir = data_path / 'cavs' / f'cavs_{date}'

    os.mkdir(activation_dir)
    os.mkdir(cav_dir)

    bottlenecks = ['encoder', 'enc_augment', 'summarizer', 'extended_classifier', 'classifier']
    alphas = [0.1]

    target =  'Left fist, performed'
    
    labels = ['Left fist, performed', 'Right fist, performed']
        
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
    results = my_tcav.run(run_parallel = False)

    # Save dictionary that also contains numpy array
    with open(data_path / f'tcav_results_{date}_{str(concept_folder)}', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


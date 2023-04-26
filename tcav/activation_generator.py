from abc import ABCMeta
from abc import abstractmethod
from multiprocessing import dummy as multiprocessing
import os.path
import numpy as np
import PIL.Image
import tensorflow as tf
import pickle
import torch
import torch.nn.functional as Ftorch

class ActivationGeneratorInterface(object):
    """Interface for an activation generator for a model"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def process_and_load_activations(self, bottleneck_names, concepts):
        pass

    @abstractmethod
    def get_model():
        pass


class ActivationGeneratorBase(ActivationGeneratorInterface):
    """Basic abstract activation generator for a model"""

    def __init__(self, model, acts_dir, max_examples=500):
        self.model = model
        self.acts_dir = acts_dir
        self.max_examples = max_examples

    def get_model(self):
        return self.model

    @abstractmethod
    def get_examples_for_concept(self, concept):
        pass

    def get_activations_for_concept(self, concept, bottleneck):
        examples = self.get_examples_for_concept(concept)
        return self.get_activations_for_examples(examples, bottleneck)

    def get_activations_for_examples(self, examples, bottleneck):
        acts = self.model.run_examples(examples, bottleneck)
        return self.model.reshape_activations(acts).squeeze()

    def process_and_load_activations(self, bottleneck_names, concepts):
        acts = {}
        if self.acts_dir and not tf.io.gfile.exists(self.acts_dir):
            tf.io.gfile.makedirs(self.acts_dir)

        for concept in concepts:
            if concept not in acts:
                acts[concept] = {}
            for bottleneck_name in bottleneck_names:
                acts_path = os.path.join(self.acts_dir, 'acts_{}_{}'.format(
                    concept, bottleneck_name)) if self.acts_dir else None
                if acts_path and tf.io.gfile.exists(acts_path):
                    with tf.io.gfile.GFile(acts_path, 'rb') as f:
                        acts[concept][bottleneck_name] = np.load(f, allow_pickle=True).squeeze()
                        tf.compat.v1.logging.info('Loaded {} shape {}'.format(
                            acts_path, acts[concept][bottleneck_name].shape))
                else:
                    acts[concept][bottleneck_name] = self.get_activations_for_concept(
                        concept, bottleneck_name)
                    if acts_path:
                        tf.compat.v1.logging.info('{} does not exist, Making one...'.format(
                            acts_path))
                        with tf.io.gfile.GFile(acts_path, 'w') as f:
                            np.save(f, acts[concept][bottleneck_name], allow_pickle=False)
        return acts

class EEGActivationGenerator(ActivationGeneratorBase):
    def __init__(self, model, source_dir, acts_dir, max_examples=10):
        self.model = model
        self.source_dir = source_dir 
        super(EEGActivationGenerator, self).__init__(
            model, acts_dir, max_examples)


    def get_examples_for_concept(self, concept):
        concept_dir = os.path.join(self.source_dir, concept)
        eeg_paths_list = [os.path.join(concept_dir, d)
                     for d in tf.io.gfile.listdir(concept_dir)]
        eeg_exp = self.load_eegs_from_files(eeg_paths_list, self.max_examples) 

        return eeg_exp

    # eeg paths concists of EEG concept files of standard length for each concept
    def load_eegs_from_files(self, eeg_paths, max_examples, do_shuffle=True):
        eeg_tensor = torch.from_numpy(np.array([]))
        count = 0 
        theLength = self.model.eeg_shape[-1]
        
        if do_shuffle:
            np.random.shuffle(eeg_paths)        
        
        for eeg in eeg_paths : 
            fileName = open(eeg, 'rb')
            
            eeg_tensor_new = pickle.load(fileName)            

            #Files should be of same size for each concept (sampl_frec*sec) but if there is some difference in how the files were saved, we standardize the file length to the length of the first file
            if(eeg_tensor_new.size(2) < theLength) : 
                diff = theLength - eeg_tensor_new.size(2) 
                zero_tensor = torch.zeros(1, 20, diff)
                eeg_tensor_new = torch.cat((eeg_tensor_new, zero_tensor), dim=2)
                
            if(eeg_tensor_new.size(2) > theLength) : 
                diff = eeg_tensor_new.size(2) - theLength 
                eeg_tensor_new = eeg_tensor_new[:, :, :-diff]
                
            eeg_tensor = torch.cat((eeg_tensor, eeg_tensor_new), 0)
            count = count +1 
            if(eeg_tensor.size(0) == max_examples) : 
                break

        return eeg_tensor.to(dtype=torch.float32)
import cav as cav
import model as model
import tcav as tcav
import tcav.utils as utils
import utils_plot as utils_plot  # utils_plot requires matplotlib
import os
import torch
import activation_generator as act_gen
import tensorflow as tf
from dn3_ext import LinearHeadBENDR
import pickle
import os

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())

# source_dir: where images of concepts, target class and random images (negative samples when learning CAVs) live. Each should be a sub-folder within this directory.
# Note that random image directories can be in any name. In this example, we are using random500_0, random500_1,.. for an arbitrary reason. You need roughly 50-200 images per concept and target class (10-20 pictures also tend to work, but 200 is pretty safe).

# cav_dir: directory to store CAVs (None if you don't want to store)

# target, concept: names of the target class (that you want to investigate) and concepts (strings) - these are folder names in source_dir

# bottlenecks: list of bottleneck names (intermediate layers in your model) that you want to use for TCAV. These names are defined in the model wrapper below.

source_dir = '/work3/s202059/TCAV_folders_60s/' 

results_dir =  '/work3/s202059/tcav_results' 

working_dir = '/work3/s202059/tcav_class_test_30' # './BENDR/tcav_class_test'
activation_dir = working_dir + '/activations/'
cav_dir = working_dir + '/cavs/'
bottlenecks = ['extended_classifier', 'summarizer', 'classifier']


utils.make_dir_if_not_exists(results_dir)
utils.make_dir_if_not_exists(working_dir)
utils.make_dir_if_not_exists(activation_dir)
utils.make_dir_if_not_exists(cav_dir)

# this is a regularizer penalty parameter for linear classifier to get CAVs.
alphas = [0.1]

target =  #'target_Seiz' #'target_left'
# concepts are stored in folders with these names
concepts = ['spsw', 'artf', 'bckg', 'gped', 'pled', 'eyem', 'musc'] # ['left_MI', 'artf', 'bckg', 'spsw', 'gped', 'bckg'] # ['left_MI', 'right_MI', 'eyem', 'musc'] #["eyem", "musc"]

labels = ['target_noSeiz', 'target_Seiz'] #['target_left', 'target_right'] #has to be in correct order, left,target_noSeiz = 0, right,target_Seiz = 1

# Use your trained saved BENDR model layers here as a '.pth' file
modelPath =  "/zhome/fb/9/153241/Desktop/Test/EEG_Thesis/BENDR/BENDR_SEIZ_Models/BENDR_SEIZ_fold_0.pth" # ".\BENDR\BENDR_pretrained\save_BENDR_model_LeftRight.pth"

nameOfConcepts = ""
for i in concepts : 
   nameOfConcepts += i

nameOfRun = 'LinearHeadBendr_seiz_allConcepts_60s'

sample_length_target = 60*256 #how long is the example in secs, times the sampling freq

mymodel = model.BENDRWrapper(labels, modelPath, sample_length_target)


act_generator = act_gen.EEGActivationGenerator(
   mymodel, source_dir, activation_dir, max_examples=25
   )


tf.compat.v1.logging.set_verbosity(0)
num_random_exp = 50  # folders (random500_0, random500_1, .., random500_50)

mytcav = tcav.TCAV(target,
                   concepts,
                   bottlenecks,
                   act_generator,
                   alphas,
                   cav_dir=cav_dir,
                   num_random_exp=num_random_exp)
                   
print('Loading mytcav')
results = mytcav.run()

nameOfFrame = '/work3/s202059/tcav_results_60/'  + nameOfRun + "_" + target + "_" + str(num_random_exp )+ "_concepts" + nameOfConcepts

with open(nameOfFrame + '.pkl', "wb") as f : 
   pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


#utils_plot.plot_results(results, num_random_exp=num_random_exp)

utils_plot.plot_results(results, num_random_exp=num_random_exp, plot_hist= True,  save_fig = True, nameOfDataFrame = nameOfFrame )

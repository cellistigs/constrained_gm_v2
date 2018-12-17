import os
from shutil import copy
#import cgm_train
#from cgm_train.costs import *
## This is a configuration file that contains relevant variables to share across the input,
## architecture, and cost, as well as in the actual training scripts

### Description: Give a high level overview of what the network does, so that you don't forget
"""
New version of vanilla VAE- automated model construction, multiple MC samples, and better name scoping. Test out loading in partial models with this as well.
"""


### ID features: names that identify folder that data from this network is stored in,
### as well as the names of data generated from it.
foldername = 'final_project_gmm'
videoname = 'final_project_gmm_video'

### Input features: at some point, maybe it would be better to write these from
### our input generation method.
imsize = 32
native_fullsize = 128
native_imageshift = 20 ## These can be aggregated
native_imagemean = 40

### Internal variables: things that we should keep track of in the course of training
### and passing inputs through our models
dim_v = imsize*imsize 
dim_z = 4 
dim_y = 10
dim_p = 8
### Training variables: variables that track parameters relevant for trianing
batch_size = 200
learning_rate = 1e-3
epsilon = 1e-10
MAX_EPOCHS = 30000
training = True
nb_channels = 1

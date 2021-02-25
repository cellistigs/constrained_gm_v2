import os
from shutil import copy
import cgm_train
from cgm_train.costs import *
## This is a configuration file that contains relevant variables to share across the input,
## architecture, and cost, as well as in the actual training scripts

### Description: Give a high level overview of what the network does, so that you don't forget
"""
A network that is designed to extend the results that we have seen that describe how
we can perform regression of the intensity shifting ball input onto a variety of different
backgrounds. We would like to know if we can indeed learn a "separable" representation as
a regression task: i.e., if it is able to predict a low dimensional code for an unseen background.
###### NOTE: YOU ACtually messed up, and wrote this same description for the decoder. Note that what
you actually want for this description is the encoder!!
"""

### ID features: names that identify folder that data from this network is stored in,
### as well as the names of data generated from it.
foldername = 'encoder_image_regress_largescale'
videoname = 'foldername'

### Input features: at some point, maybe it would be better to write these from
### our input generation method.
imsize = 64
native_fullsize = 128
native_imageshift = 20 ## These can be aggregated
native_imagemean = 40

### Internal variables: things that we should keep track of in the course of training
### and passing inputs through our models
dim_v = 1
dim_z = 2

### Training variables: variables that track parameters relevant for trianing
batch_size = 50
learning_rate = 1e-7
epsilon = 1e-10
MAX_EPOCHS = 100000

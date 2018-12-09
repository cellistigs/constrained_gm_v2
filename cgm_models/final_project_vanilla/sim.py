import tensorflow as tf
import numpy as np
import imageio
from skimage.transform import resize
from scipy import misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import cgm_train
from cgm_train.costs import *
from cgm_train.models_basetf import *
from cgm_train.input import *
from config import native_fullsize,learning_rate,epsilon,MAX_EPOCHS,imsize,batch_size,videoname

## We are going to simulate from the same dataset that we trained.
## Load in the data:
filenames = ['../../data/mother_true.tfrecords']
ims,position,mouse,video,initializer = VAE_pipeline(filenames,batch_size,imsize)

## Push it through the network:
out,mean,logstd = VAE_vanilla_graph(ims,dim_z,'vanilla_graph',training=False)

var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vanilla_graph')
# var_list is important. it sees the tensorflow variables, that are in the scope of the first_net in this default graph.
saver = tf.train.Saver(var_list = var_list)
checkpointdirectory = videoname
print(var_list)
init = tf.global_variables_initializer()
epoch = 1181
with tf.Session() as sess:
    sess.run(init)
    sess.run(initializer)
    saver.restore(sess,checkpointdirectory+'/modelep'+str(epoch)+'.ckpt')
    sess.run([out,mean,logstd])

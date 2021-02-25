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
#filenames = ['../../data/mother_true.tfrecords']
filenames = ['../../../../../Blei/mother_test.tfrecords']
ims,position,mouse,video,initializer = VAE_pipeline(filenames,batch_size,imsize)

is_training = tf.placeholder(dtype = tf.int32)

## Push it through the network:
out,mean,logstd = VAE_vanilla_graph(ims,dim_z,'vanilla_graph',training=True,nb_samples =1)

var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vanilla_graph')
# var_list is important. it sees the tensorflow variables, that are in the scope of the first_net in this default graph.
saver = tf.train.Saver(var_list = var_list)
checkpointdirectory = videoname
print(var_list)
init = tf.global_variables_initializer()
epoch = 20
with tf.Session() as sess:
    sess.run(init)
    sess.run(initializer)
    saver.restore(sess,checkpointdirectory+'/modelep'+str(epoch)+'.ckpt')
    rec,image,mean,_ = sess.run([ims,out,mean,logstd],feed_dict = {is_training:False})
    for i in range(16):
        plt.plot(mean[:,i])
    plt.savefig("means")
    for i in range(batch_size):
        fig,ax = plt.subplots(2,)
        ax[0].imshow(image[0,i,:,:,0])
        ax[1].imshow(rec[i,:,:,0]/255.)
        plt.savefig('testimage'+str(i))
        plt.close()

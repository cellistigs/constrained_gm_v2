import tensorflow as tf
import sys
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

## Load in the data:
filenames = ['../../data/mother_test.tfrecords']
ims,position,mouse,video,initializer = VAE_pipeline(filenames,batch_size,imsize)
ims = ims/255.

is_training = tf.placeholder(dtype = tf.int32)

## Push it through the network:
out,mean,logstd = VAE_vanilla_graph(ims,dim_z,'vanilla_graph',training=is_training)

load = False
if load == True:
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vanilla_graph')
    # var_list is important. it sees the tensorflow variables, that are in the scope of the first_net in this default graph.
    saver = tf.train.Saver(var_list = var_list)
    checkpointdirectory = videoname
## Calculate the cost:
ll = VAE_likelihood_MC(ims,out)
kl = D_kl_prior_cost(mean,logstd)

full_elbo = ll+kl

## Define an optimizer:
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate,epsilon=epsilon).minimize(-full_elbo)

## Now run iterative training:
print('Running Tensorflow model')
checkpointdirectory = videoname
init = tf.global_variables_initializer()
if not os.path.exists(checkpointdirectory):
    os.mkdir(checkpointdirectory)

# #Add iterator to the list of saveable objects:
#
# saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
#
# # Save the iterator state by adding it to the saveable objects collection.
# tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
losses = []
saver = tf.train.Saver(max_to_keep=2)
epoch = 57
with tf.Session() as sess:
    sess.run(init)
    if load == True:
        saver.restore(sess,checkpointdirectory+'/modelep'+str(epoch)+'.ckpt')
    max_epochs = MAX_EPOCHS
    scale = 1.0
    for epoch in range(max_epochs):
        print(epoch)
        sess.run(initializer)
        epoch_cost = 0
        i = 0
        while True:
            try:
                progress = i/(1000*len(filenames)/(batch_size))*100
                sys.stdout.write("Train progress: %d%%   \r" % (progress) )
                sys.stdout.flush()
                _,cost = sess.run([optimizer,full_elbo],feed_dict={is_training:1})
                epoch_cost+=cost
                i+=1
            except tf.errors.OutOfRangeError:
                break
        if epoch % 200 == 0:
            fig,ax = plt.subplots(2,3)
            ax[0,0].imshow(output[0,0,:,:,0])
            ax[1,0].imshow(gt[0,:,:,0])
            ax[0,1].imshow(output[0,1,:,:,0])
            ax[1,1].imshow(gt[1,:,:,0])
            ax[0,2].imshow(output[0,2,:,:,0])
            ax[1,2].imshow(gt[2,:,:,0])
            plt.savefig(checkpointdirectory+'/check_epoch'+str(epoch))
            plt.close()
        losses.append(epoch_cost)
        print('Loss for epoch '+str(epoch)+': '+str(epoch_cost))
        save_path = saver.save(sess,checkpointdirectory+'/modelep'+str(epoch)+'.ckpt')
        if epoch%100 == 0:
            np.save(checkpointdirectory+'/loss',losses)

import tensorflow as tf
import sys
import numpy as np
from scipy import misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import cgm_train
from cgm_train.costs import *
from cgm_train.models_basetf import *
from cgm_train.input import *
from config import native_fullsize,learning_rate,epsilon,MAX_EPOCHS,imsize,batch_size,videoname,dim_z,dim_y,nb_channels

## Gaussian mixture prior deep generative model.
## Load in the weights from the vanilla model where appropriate to speed training.

## Load in the data:
filenames = ['../../data/virgin_train.tfrecords']
filenames_test = ['../../data/virgin_val.tfrecords']
ims,position,mouse,video,initializer,test_initializer = VAE_traintest_pipeline(filenames,filenames_test,batch_size,imsize)
ims = ims/255.
nb_samples = 5
position = tf.cast(position,tf.float32)
position = tf.reshape(position,(batch_size,dim_p))
is_training = tf.placeholder(dtype = tf.int32)
"""
Eventually, we will package what is between the hashes into a function. However,
it's easier right now to have the parts bare because we are incorporating weights
from other pretrained models.
"""
######################################################
## Our model has multiple parts:
## RECOGNITION NETWORK
# Define the part of the recognition network q(z|x,y) that inherits weights from the vanilla graph:
pure_img_codes = recog_model_imageprocess(ims,dim_z,'vanilla_graph/recog')

# Define the part of the recognition network q(z|x,y) that divides out by category:
inference_means,inference_logstds = recog_model_mixture(pure_img_codes,dim_y,dim_z,'gmm_graph/recog_m')
## Note: the returned values here are of size ((batch_size*dim_y),dimz), the result of the same (batch,dimz) size samples, that
## have been transformed by conditioning on a different category label for each. They are organized via tile on the
## actual codes, not repeat (all of category 1 first, all of cat 2, etc.)

# Define the part of the recognition network q(y|x) that gives category inferences:
inference_cats = recog_model_cat_sup(ims,position,dim_y,'gmm_graph/recog_c')
## This is of shape (batch_size,dim_y)

## Protect against zeros in categorical loss, further computations.
inference_cats = tf.clip_by_value(inference_cats,1e-7,1.0)
##

## For the sake of computation, we want to organize as 1.....b......y1....yb
inference_cats_batch = tf.reshape(tf.transpose(inference_cats),(batch_size*dim_y,-1))
## GENERATOR NETWORK

# Define the part of the generator network p(z|y)

generative_means = tf.Variable(tf.random_normal(shape=[dim_y,dim_z],mean=0.,stddev=1./np.sqrt(dim_y)),name = 'gener_means')
generative_logstds = tf.Variable(tf.random_normal(shape=[dim_y,dim_z],mean=0.,stddev=1./np.sqrt(dim_y)),name = 'gener_logstds')
## each of size (batch_size*dim_y,dim_z)
generative_means = tf.reshape(tf.tile(generative_means,(1,batch_size)),(batch_size*dim_y,dim_z))
generative_logstds = tf.reshape(tf.tile(generative_logstds,(1,batch_size)),(batch_size*dim_y,dim_z))
## each of size (batch_size*dim_y,dim_z)

## Reparametrization trick:
mean_broadcast = tf.tile(tf.expand_dims(inference_means,0),(nb_samples,1,1))
std_broadcast = tf.tile(tf.expand_dims(tf.exp(inference_logstds),0),(nb_samples,1,1))

# Sample noise:
eps = tf.random_normal((nb_samples,batch_size*dim_y,dim_z))

samples = mean_broadcast+std_broadcast*eps

samples_reshape = tf.reshape(samples,(nb_samples*batch_size*dim_y,dim_z))

# Generator network for images:
out = gener_model_vanilla(samples_reshape,'vanilla_graph/gener')

pose_prop = tf.reshape(gener_model_pose(samples_reshape),(nb_samples,batch_size*dim_y,-1))

out_reshape = tf.reshape(out,(nb_samples,batch_size*dim_y,imsize,imsize,nb_channels))

load = False
if load == True:
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vanilla_graph')
    # var_list is important. it sees the tensorflow variables, that are in the scope of the first_net in this default graph.
    saver = tf.train.Saver(var_list = var_list)
    checkpointdirectory = '../final_project_vanilla/final_project_vanilla_video'
#####################################################################
# We are evaluating on out, generative/inference means/logstds, and inference cats
## Render cost:
print(pose_prop,samples_reshape,'shape of propped')
ll = GMVAE_likelihood_MC(ims,out_reshape,inference_cats_batch)
ll_p = GMVAE_likelihood_MC_pose(position,pose_prop,inference_cats_batch)
kl_c = GMVAE_cat_kl(inference_cats)
kl_g,_,_,_ = GMVAE_gauss_kl(inference_means,inference_logstds,generative_means,generative_logstds,inference_cats_batch)
# lm = GMVAE_cluster_cost(samples,generative_means,generative_logstds,inference_cats_batch)
# lp = GMVAE_prior_cost(inference_cats_batch)

# hg = GMVAE_normal_entropy(inference_logstds,inference_cats_batch)
# hc = GMVAE_cat_entropy(inference_cats)

full_elbo = ll+kl_c+kl_g+ll_p

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
saver_newsave = tf.train.Saver(max_to_keep=2)
epoch = 43
init = tf.global_variables_initializer()

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
        losses.append(epoch_cost)
        print('Loss for epoch '+str(epoch)+': '+str(epoch_cost))
        save_path = saver_newsave.save(sess,checkpointdirectory+'/modelep'+str(epoch)+'.ckpt')
        if epoch%100 == 0:
            np.save(checkpointdirectory+'/loss',losses)

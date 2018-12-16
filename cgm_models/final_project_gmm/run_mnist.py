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
from config import native_fullsize,learning_rate,epsilon,MAX_EPOCHS,imsize,batch_size,videoname,dim_z,dim_y,nb_channels

## Toy models for testing base functionality. This is in mnist.

## Load in the data:
train, test = tf.keras.datasets.mnist.load_data()
mnist_x, mnist_y = train
dataset = tf.data.Dataset.from_tensor_slices(mnist_x[:12000,:,:])


## We're going to use an even simpler dataset: just 4 gaussian mixtures in a 16-d space. 
#mean_vecs = [((i+np.arange(16))%(4) == 0).astype(int) for i in range(4)]

#all_vecdata = []

#for vec in mean_vecs:
#    vec_data = tf.clip_by_value(tf.random_normal(mean = vec,stddev = 0.05,shape = (1000,16)),0,1)
#    all_vecdata.append(vec_data)

#interleaved = tf.reshape(tf.stack(all_vecdata, axis=1),[-1, tf.shape(all_vecdata[0])[1]])

#interleaved_ims = tf.reshape(interleaved,(4*1000,4,4))

#dataset = tf.data.Dataset.from_tensor_slices(interleaved_ims)

# dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

batched = dataset.shuffle(batch_size*2).apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

iterator = tf.data.Iterator.from_structure(batched.output_types,batched.output_shapes)

initializer = iterator.make_initializer(batched)

ims = iterator.get_next()

ims = tf.image.resize_images(tf.expand_dims(ims,-1),[imsize,imsize])/255.

nb_samples = 1
is_training = tf.placeholder(dtype = tf.int32)

######################################################
## Our model has multiple parts:
## RECOGNITION NETWORK
# Define the part of the recognition network q(z|x,y) that inherits weights from the vanilla graph:
pure_img_codes = recog_model_imageprocess(ims,dim_z,'vanilla_graph/recog',bn = True )

# Define the part of the recognition network q(z|x,y) that divides out by category:
inference_means,inference_logstds = recog_model_mixture(pure_img_codes,dim_y,dim_z,'gmm_graph/recog_m')
## Note: the returned values here are of size ((batch_size*dim_y),dimz), the result of the same (batch,dimz) size samples, that
## have been transformed by conditioning on a different category label for each. They are organized via tile on the
## actual codes, not repeat (all of category 1 first, all of cat 2, etc.)

# Define the part of the recognition network q(y|x) that gives category inferences:
inference_cats = recog_model_cat(ims,dim_y,'gmm_graph/recog_c',bn = True)
## This is of shape (batch_size,dim_y)

## Protect against zeros in categorical loss, further computations.
inference_cats = tf.clip_by_value(inference_cats,1e-7,1.0)
##

## For the sake of computation, we want to organize as 1.....b......y1....yb
inference_cats_batch = tf.reshape(tf.transpose(inference_cats),(batch_size*dim_y,-1))
## GENERATOR NETWORK

# Define the part of the generator network p(z|y)

#generative_means,generative_logstds = gener_model_mixture(inference_cats_batch,dim_z,'gmm_graph/gener_c')
generative_means = tf.Variable(tf.random_normal(shape=[dim_y,dim_z],mean=0.,stddev=1./np.sqrt(dim_y)),name = 'gener_means')
generative_logstds = tf.Variable(tf.random_normal(shape=[dim_y,dim_z],mean=0.,stddev=1./np.sqrt(dim_y)),name = 'gener_logstds')
## each of size (batch_size*dim_y,dim_z)
generative_means = tf.reshape(tf.tile(generative_means,(1,batch_size)),(batch_size*dim_y,dim_z))
generative_logstds = tf.reshape(tf.tile(generative_logstds,(1,batch_size)),(batch_size*dim_y,dim_z))
## Reparametrization trick:

mean_broadcast = tf.tile(tf.expand_dims(inference_means,0),(nb_samples,1,1))
std_broadcast = tf.tile(tf.expand_dims(tf.exp(inference_logstds),0),(nb_samples,1,1))

# Sample noise:
eps = tf.random_normal((nb_samples,batch_size*dim_y,dim_z))

samples = mean_broadcast+std_broadcast

samples_reshape = tf.reshape(samples,(nb_samples*batch_size*dim_y,dim_z))

# Generator network for images:
out = gener_model_vanilla(samples_reshape,'vanilla_graph/gener')

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
ll = GMVAE_likelihood_MC(ims,out_reshape,inference_cats_batch)
kl_c = GMVAE_cat_kl(inference_cats)
kl_g = GMVAE_gauss_kl(inference_means,inference_logstds,generative_means,generative_logstds,inference_cats_batch)

alpha = tf.placeholder(dtype = float,name = 'alpha')

full_elbo = ll+alpha*(kl_c+kl_g)

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
                progress = i/(1000/(batch_size))*100
                sys.stdout.write("Train progress: %d%%   \r" % (progress) )
                sys.stdout.flush()
                _,cost,output,e = sess.run([optimizer,full_elbo,out_reshape,inference_cats],feed_dict={is_training:1,alpha:1})
                epoch_cost+=cost
                i+=1
            except tf.errors.OutOfRangeError:
                break
            #print((np.min(a),np.min(b),np.min(c),np.min(d),np.min(e)))
        print(e[:3,:])
        if epoch % 20 == 0:
            fig,ax = plt.subplots(3,)
            print(output.shape)
            ax[0].imshow(output[0,0,:,:,0])
            ax[1].imshow(output[0,1,:,:,0])
            ax[2].imshow(output[0,2,:,:,0])
            plt.savefig(checkpointdirectory+'/check_epoch'+str(epoch))
            plt.close()
        losses.append(epoch_cost)
        print('Loss for epoch '+str(epoch)+': '+str(epoch_cost))
        save_path = saver_newsave.save(sess,checkpointdirectory+'/modelep_mnist'+str(epoch)+'.ckpt')
        if epoch%100 == 0:
            np.save(checkpointdirectory+'/loss',losses)

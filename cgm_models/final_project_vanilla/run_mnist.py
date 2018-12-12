import tensorflow as tf
import sys
import numpy as np
import imageio
from skimage.transform import resize
from sklearn import datasets
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

## Toy models for testing base functionality. This is in mnist.

## Load in the data:
from sklearn import datasets
digits = datasets.load_digits()
digit_data = digits.data
## Reshape:
digit_data = digit_data.reshape(-1,imsize,imsize).astype(np.float32)
## Add Channel:
digit_data = digit_data[:,:,:,np.newaxis].repeat(3,axis=-1)
## Normalize:
digit_data = digit_data/np.max(digit_data)

dataset = tf.data.Dataset.from_tensor_slices(digit_data)

dataset.shuffle(batch_size*2).apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

batched = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

iterator = tf.data.Iterator.from_structure(batched.output_types,batched.output_shapes)

initializer = iterator.make_initializer(batched)

ims = iterator.get_next()

## Push it through the network:
out,mean,logstd = VAE_vanilla_graph(ims,dim_z,'vanilla_graph',training=True,nb_samples = 1)


load = False
if load == True:
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vanilla_graph')
    # var_list is important. it sees the tensorflow variables, that are in the scope of the first_net in this default graph.
    saver = tf.train.Saver(var_list = var_list)
    checkpointdirectory = videoname
## Calculate the cost:
ll = VAE_likelihood_MC(ims,out,1)
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
epoch = 11880
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
                progress = i/(1000*1/(batch_size))*100
                sys.stdout.write("Train progress: %d%%   \r" % (progress) )
                sys.stdout.flush()
                _,cost,lik,cro = sess.run([optimizer,full_elbo,ll,kl])
                epoch_cost+=cost
                i+=1
            except tf.errors.OutOfRangeError:
                break
        print(lik,cro)
        # gt_expand = tf.tile(tf.expand_dims(gt,0),(nb_samples,1,1,1,1))
        # for i in range(5):
        #     for j in range(5):
        #         fig,ax = plt.subplots(2,1)
        #         ax[0].imshow(gt_expand[i,j,:,:,:])
        #         ax[1].imshow(output[i,j,:,:,:])
        #         ax[0].set_title('orig'+'sample'+str(i)+'img')
        #         ax[1].set_title('reconst')

        losses.append(epoch_cost)
        print('Loss for epoch '+str(epoch)+': '+str(epoch_cost))
        save_path = saver.save(sess,checkpointdirectory+'/modelep_mnist'+str(epoch)+'.ckpt')
        if epoch%100 == 0:
            np.save(checkpointdirectory+'/loss',losses)

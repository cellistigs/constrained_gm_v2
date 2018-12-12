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
## Load in the data:
train, test = tf.keras.datasets.mnist.load_data()
mnist_x, mnist_y = train
dataset = tf.data.Dataset.from_tensor_slices(mnist_x[:2000,:,:])

dataset.shuffle(batch_size*2).apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

batched = dataset.shuffle(batch_size*2).apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

iterator = tf.data.Iterator.from_structure(batched.output_types,batched.output_shapes)

initializer = iterator.make_initializer(batched)

ims = iterator.get_next()

ims = tf.image.resize_images(tf.expand_dims(ims,-1),[imsize,imsize])/255.

## Push it through the network:
out,mean,logstd = VAE_vanilla_graph(ims,dim_z,'vanilla_graph',training=False,nb_samples = 1,sample_mean = True)

var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vanilla_graph')
# var_list is important. it sees the tensorflow variables, that are in the scope of the first_net in this default graph.
saver = tf.train.Saver(var_list = var_list)
checkpointdirectory = videoname
print(var_list)
init = tf.global_variables_initializer()
epoch = 69
with tf.Session() as sess:
    sess.run(init)
    sess.run(initializer)
    saver.restore(sess,checkpointdirectory+'/modelep_mnist'+str(epoch)+'.ckpt')
    rec,image,mean,_ = sess.run([ims,out,mean,logstd])
    input_expand = np.tile(rec[np.newaxis,:,:,:,:],(5,1,1,1,1))
    for i in range(16):
        plt.plot(mean[:,i])
    plt.savefig("means")
    for i in range(batch_size):
        for j in range(1):
            fig,ax = plt.subplots(2,)
            ax[0].imshow(image[j,i,:,:,0])
            ax[1].imshow(input_expand[j,i,:,:,0])
            plt.savefig('testimage'+str(i)+'sample'+str(j))
            plt.close()

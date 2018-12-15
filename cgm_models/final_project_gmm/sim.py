## Simulate from the trained network
import os
import glob
import tensorflow as tf
import numpy as np
import imageio
from skimage.transform import resize
from scipy import misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import cgm_train
from cgm_train.costs import *
from cgm_train.models import *
from cgm_train.input import *
from config import native_fullsize,learning_rate,epsilon,MAX_EPOCHS,videoname

## Initialize tensorflow session:
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

## Reload the graph:
data = 'modelep298.ckpt'
metadata = data+'.meta'
saver = tf.train.import_meta_graph(videoname+'/'+metadata)
saver.restore(sess,videoname+'/'+data)
print([n.name for n in tf.get_default_graph().as_graph_def().node])
## Name vars to restore:
graph = tf.get_default_graph()
is_training = graph.get_tensor_by_name('Placeholder:0')
out_im = graph.get_tensor_by_name('vanilla_graph/gener/layer6/adjconv/Sigmoid:0')
initializer = graph.get_operation_by_name('init_op')

while True:
    try:
        # out = sess.run([pos_it,next_images_it,label_it])
        # label = int(out[2]/250)
        sess.run(initializer)
        # gen_images = sess.run(out_im,feed_dict = {pos_gt:out[0].reshape(batch_size,2)})
        image_outputs = sess.run(out_im,feed_dict={is_training:0})
        gen_images = image_outputs
        # ball = image_outputs[1]
        # back = image_outputs[2]
    except tf.errors.OutOfRangeError:
        break
    print(image_outputs.shape)
    plt.imshow(image_outputs[0,:,:,:])
    plt.savefig('test_newsim_gmm')

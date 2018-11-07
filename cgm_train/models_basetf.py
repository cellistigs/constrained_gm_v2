import sys
import numpy as np
import tensorflow as tf
from config import batch_size,imsize,dim_z,dim_v

### Take out the residual and look at a simple linear sum at the end:
def gener_model_dyn_full_grid_background_additive(grid,background):
    '''This is a complicated architecture that will merge the grid and scalar outputs at an
    appropriate output point.
    '''
    background_0 = tf.layers.dense(background,1*1*512,activation=None,kernel_initializer=tf.random_uniform_initializer(0.1))
    background_1 = tf.layers.conv2d_transpose(tf.reshape(background_0,shape =(-1,1,1,512)),256,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    background_2 = tf.layers.conv2d_transpose(background_1,128,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    background_3 = tf.layers.conv2d_transpose(background_2,64,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    background_4 = tf.layers.conv2d_transpose(background_3,32,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    background_5 = tf.layers.conv2d_transpose(background_4,16,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    background_6 = tf.layers.conv2d_transpose(background_5,5,5,strides = 2,padding = 'same',activation = tf.nn.sigmoid,name = 'lastback')


    grid_0 = tf.layers.conv2d_transpose(grid,32,5,strides = 1,padding = 'same',activation = tf.nn.elu)
    grid_1 = tf.layers.conv2d_transpose(grid_0,32,5,strides = 1,padding = 'same',activation = tf.nn.elu)
    grid_2 = tf.layers.conv2d_transpose(grid_1,32,5,strides = 1,padding = 'same',activation = tf.nn.elu)
    grid_3 = tf.layers.conv2d_transpose(grid_2,16,5,strides = 1,padding = 'same',activation = tf.nn.elu)
    grid_4 = tf.layers.conv2d_transpose(grid_3,16,5,strides = 1,padding = 'same',activation = tf.nn.elu)
    grid_5 = tf.layers.conv2d_transpose(grid_4,5,5,strides = 1,padding = 'same',activation = tf.nn.sigmoid,name = 'lastball')


    # added = tf.reshape(tf.add(grid_5,background_6,name = 'unconvolved'),[batch_size,imsize*imsize*5])
    added = tf.nn.sigmoid(tf.add(grid_5,background_6,name = 'unconvolved'))
    # full_0 = tf.layers.dense(added,imsize*imsize*5,activation = tf.nn.sigmoid)
    # full_final_reshaped = tf.reshape(full_0,[batch_size,imsize,imsize,5])

    # (pt.wrap(together). ## Now the scalar input and the grid is merged
    #         deconv2d(5,16,stride = 2).
    #         deconv2d(5,5,stride = 2,activation_fn=tf.nn.sigmoid)).tensor # 32x32x4 ## Now ask to regenerate the meshgrid as well.
    return added

def gener_model_dyn_full_grid_background_fullyconnected(grid,scalar,background):
    '''This is a complicated architecture that will merge the grid and scalar outputs at an
    appropriate output point.
    '''
    background_0 = tf.layers.dense(background,1*1*512,activation=None,kernel_initializer=tf.random_uniform_initializer(0.1))
    background_1 = tf.layers.conv2d_transpose(tf.reshape(background_0,shape =(-1,1,1,512)),256,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    background_2 = tf.layers.conv2d_transpose(background_1,128,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    background_3 = tf.layers.conv2d_transpose(background_2,64,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    background_4 = tf.layers.conv2d_transpose(background_3,32,5,strides = 2,padding = 'same',activation = tf.nn.elu)


    scalar_0 = tf.layers.dense(scalar,1*1*512,activation=None,kernel_initializer=tf.random_uniform_initializer(0.1))
    scalar_1 = tf.layers.conv2d_transpose(tf.reshape(scalar_0,shape =(-1,1,1,512)),256,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_2 = tf.layers.conv2d_transpose(scalar_1,128,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_3 = tf.layers.conv2d_transpose(scalar_2,64,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_4 = tf.layers.conv2d_transpose(scalar_3,32,5,strides = 2,padding = 'same',activation = tf.nn.elu)

    grid_0 = tf.layers.conv2d_transpose(grid,16,5,strides = 1,padding = 'same',activation = tf.nn.elu)
    grid_1 = tf.layers.conv2d_transpose(grid_0,16,5,strides = 1,padding = 'same',activation = tf.nn.elu)

    together = tf.concat((scalar_4,background_4,grid_1),axis = -1)

    full_0 = tf.layers.conv2d_transpose(together,16,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    full_1 = tf.layers.conv2d_transpose(full_0,5,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    full_final = tf.layers.dense(full_1,imsize*imsize*5,activation = tf.nn.sigmoid)
    full_final_reshaped = tf.reshape(full_final,shape = (None,imsize,imsize,5))

    # (pt.wrap(together). ## Now the scalar input and the grid is merged
    #         deconv2d(5,16,stride = 2).
    #         deconv2d(5,5,stride = 2,activation_fn=tf.nn.sigmoid)).tensor # 32x32x4 ## Now ask to regenerate the meshgrid as well.
    return full_final_reshaped

def gener_model_dyn_full_grid_basic(grid,scalar):
    '''This is a complicated architecture that will merge the grid and scalar outputs at an
    appropriate output point.
    '''
    scalar_0 = tf.layers.dense(scalar,1*1*512,activation=None,kernel_initializer=tf.random_uniform_initializer(0.1))
    scalar_1 = tf.layers.conv2d_transpose(tf.reshape(scalar_0,shape =(-1,1,1,512)),256,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_2 = tf.layers.conv2d_transpose(scalar_1,128,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_3 = tf.layers.conv2d_transpose(scalar_2,64,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_4 = tf.layers.conv2d_transpose(scalar_3,32,5,strides = 2,padding = 'same',activation = tf.nn.elu)

    grid_0 = tf.layers.conv2d_transpose(grid,16,5,strides = 1,padding = 'same',activation = tf.nn.elu)
    grid_1 = tf.layers.conv2d_transpose(grid_0,16,5,strides = 1,padding = 'same',activation = tf.nn.elu)

    together = tf.concat((scalar_4,grid_1),axis = -1)

    full_0 = tf.layers.conv2d_transpose(together,16,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    full_1 = tf.layers.conv2d_transpose(full_0,5,5,strides = 2,padding = 'same',activation = tf.nn.elu)

    # (pt.wrap(together). ## Now the scalar input and the grid is merged
    #         deconv2d(5,16,stride = 2).
    #         deconv2d(5,5,stride = 2,activation_fn=tf.nn.sigmoid)).tensor # 32x32x4 ## Now ask to regenerate the meshgrid as well.
    return full_final_reshaped

## fullsize inputs with residuals.
def gener_model_dyn_fullsize_grid(grid,scalar):
    '''This is a complicated architecture that will merge the grid and scalar outputs at an
    appropriate output point.
    '''
    scalar_0 = tf.layers.dense(scalar,1*1*512,activation=None,kernel_initializer=tf.random_uniform_initializer(0.1))
    scalar_1 = tf.layers.conv2d_transpose(tf.reshape(scalar_0,shape =(-1,1,1,512)),256,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_2 = tf.layers.conv2d_transpose(scalar_1,128,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_3 = tf.layers.conv2d_transpose(scalar_2,64,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_4 = tf.layers.conv2d_transpose(scalar_3,32,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_5 = tf.layers.conv2d_transpose(scalar_4,32,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_6 = tf.layers.conv2d_transpose(scalar_6,32,5,strides = 2,padding = 'same',activation = tf.nn.elu)

    grid_0 = tf.layers.conv2d_transpose(grid,16,5,strides = 1,padding = 'same',activation = tf.nn.elu)
    grid_1 = tf.layers.conv2d_transpose(grid_0,16,5,strides = 1,padding = 'same',activation = tf.nn.elu)

    together = tf.concat((scalar_6,grid_1),axis = -1)

    full_0 = tf.layers.conv2d_transpose(together,16,5,strides = 1,padding = 'same',activation = tf.nn.elu)
    full_1 = tf.layers.conv2d_transpose(full_0,5,5,strides = 1,padding = 'same',activation = tf.nn.elu)

    # (pt.wrap(together). ## Now the scalar input and the grid is merged
    #         deconv2d(5,16,stride = 2).
    #         deconv2d(5,5,stride = 2,activation_fn=tf.nn.sigmoid)).tensor # 32x32x4 ## Now ask to regenerate the meshgrid as well.
    return full_final_reshaped

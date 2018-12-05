import sys
import numpy as np
import tensorflow as tf
from config import batch_size,imsize,dim_z,dim_v,training

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
    scalar_1 = tf.layers.conv2d_transpose(tf.reshape(scalar_0,shape =(-1,1,1,512)),256,5,strides = 2,padding = 'same',activation = tf.nn.sigmoid)
    scalar_2 = tf.layers.conv2d_transpose(scalar_1,128,1,strides = 2,padding = 'same',activation = tf.nn.sigmoid )
    scalar_3 = tf.layers.conv2d_transpose(scalar_2,64,2,strides = 2,padding = 'same',activation = tf.nn.sigmoid)
    scalar_4 = tf.layers.conv2d_transpose(scalar_3,32,3,strides = 2,padding = 'same',activation = tf.nn.sigmoid)
    scalar_5 = tf.layers.conv2d_transpose(scalar_4,16,4,strides = 2,padding = 'same',activation = tf.nn.sigmoid)
    scalar_6 = tf.layers.conv2d_transpose(scalar_5,8,5,strides = 2,padding = 'same',activation = tf.nn.sigmoid)

    grid_0 = tf.layers.conv2d_transpose(grid,8,3,strides = 1,padding = 'same',activation = tf.nn.sigmoid)
    grid_1 = tf.layers.conv2d_transpose(grid_0,8,3,strides = 1,padding = 'same',activation = tf.nn.sigmoid)

    together = tf.concat((scalar_6,grid_1),axis = -1)

    full_0 = tf.layers.conv2d_transpose(together,16,5,strides = 1,padding = 'same',activation = tf.nn.elu)
    full_1 = tf.layers.conv2d_transpose(full_0,5,5,strides = 1,padding = 'same',activation = tf.nn.sigmoid,name = 'netout')

    # (pt.wrap(together). ## Now the scalar input and the grid is merged
    #         deconv2d(5,16,stride = 2).
    #         deconv2d(5,5,stride = 2,activation_fn=tf.nn.sigmoid)).tensor # 32x32x4 ## Now ask to regenerate the meshgrid as well.
    return full_1

def gener_model_dyn_fullsize_grid_aux(grid,scalar):
    '''This is a complicated architecture that will merge the grid and scalar outputs at an
    appropriate output point.
    '''
    scalar_0 = tf.layers.dense(scalar,1*1*512,activation=None,kernel_initializer=tf.random_uniform_initializer(0.1))
    scalar_1 = tf.layers.conv2d_transpose(tf.reshape(scalar_0,shape =(-1,1,1,512)),256,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_1_norm = tf.layers.batch_normalization(scalar_1, training=training)
    scalar_2 = tf.layers.conv2d_transpose(scalar_1_norm,128,1,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_2_norm = tf.layers.batch_normalization(scalar_2, training=training)
    scalar_3 = tf.layers.conv2d_transpose(scalar_2_norm,64,2,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_3_norm = tf.layers.batch_normalization(scalar_3, training=training)
    scalar_4 = tf.layers.conv2d_transpose(scalar_3_norm,32,3,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_4_norm = tf.layers.batch_normalization(scalar_4, training=training)
    scalar_5 = tf.layers.conv2d_transpose(scalar_4_norm,16,4,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_5_norm = tf.layers.batch_normalization(scalar_5, training=training)
    scalar_6 = tf.layers.conv2d_transpose(scalar_5_norm,8,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_6_norm = tf.layers.conv2d_transpose(scalar_5_norm,8,5,strides = 2,padding = 'same',activation = tf.nn.elu)

    grid_0 = tf.layers.conv2d_transpose(grid,8,3,strides = 1,padding = 'same',activation = tf.nn.elu)
    grid_0_norm = tf.layers.batch_normalization(grid_0, training = training)
    grid_1 = tf.layers.conv2d_transpose(grid_0_norm,8,3,strides = 1,padding = 'same',activation = tf.nn.elu)
    grid_1_norm = tf.layers.batch_normalization(grid_1, training = training)

    together = tf.concat((scalar_6_norm,grid_1_norm),axis = -1)

    # auxiliary = tf.layers.dense(tf.reshape(together,shape = (batch_size,imsize*imsize*16)),2,activation= None,name = 'netaux')

    full_0 = tf.layers.conv2d_transpose(together,16,5,strides = 1,padding = 'same',activation = tf.nn.elu)
    full_0_norm = tf.layers.batch_normalization(full_0, training = training)
    full_1 = tf.layers.conv2d_transpose(full_0_norm,5,5,strides = 1,padding = 'same',activation = tf.nn.sigmoid,name = 'netout')

    # (pt.wrap(together). ## Now the scalar input and the grid is merged
    #         deconv2d(5,16,stride = 2).
    #         deconv2d(5,5,stride = 2,activation_fn=tf.nn.sigmoid)).tensor # 32x32x4 ## Now ask to regenerate the meshgrid as well.
    return full_1

def gener_model_dyn_fullsize_grid_colors(grid,scalar,params):
    '''This is a complicated architecture that will merge the grid and scalar outputs at an
    appropriate output point.
    '''
    scalar_0 = tf.layers.dense(scalar,1*1*512,activation=None,kernel_initializer=tf.random_uniform_initializer(0.1))
    scalar_1 = tf.layers.conv2d_transpose(tf.reshape(scalar_0,shape =(-1,1,1,512)),256,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_1_norm = tf.layers.batch_normalization(scalar_1, training=training)
    scalar_2 = tf.layers.conv2d_transpose(scalar_1_norm,128,1,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_2_norm = tf.layers.batch_normalization(scalar_2, training=training)
    scalar_3 = tf.layers.conv2d_transpose(scalar_2_norm,64,2,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_3_norm = tf.layers.batch_normalization(scalar_3, training=training)
    scalar_4 = tf.layers.conv2d_transpose(scalar_3_norm,32,3,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_4_norm = tf.layers.batch_normalization(scalar_4, training=training)
    scalar_5 = tf.layers.conv2d_transpose(scalar_4_norm,16,4,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_5_norm = tf.layers.batch_normalization(scalar_5, training=training)
    scalar_6 = tf.layers.conv2d_transpose(scalar_5_norm,8,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    scalar_6_norm = tf.layers.batch_normalization(scalar_6, training=training)

    grid_0 = tf.layers.conv2d_transpose(grid,8,3,strides = 1,padding = 'same',activation = tf.nn.elu)
    grid_0_norm = tf.layers.batch_normalization(grid_0, training = training)
    grid_1 = tf.layers.conv2d_transpose(grid_0_norm,8,3,strides = 1,padding = 'same',activation = tf.nn.elu)
    grid_1_norm = tf.layers.batch_normalization(grid_1, training = training)


    together = tf.concat((scalar_6_norm,grid_1_norm),axis = -1)

    # auxiliary = tf.layers.dense(tf.reshape(together,shape = (batch_size,imsize*imsize*16)),2,activation= None,name = 'netaux')

    full_0 = tf.layers.conv2d_transpose(together,16,5,strides = 1,padding = 'same',activation = tf.nn.elu)
    full_0_norm = tf.layers.batch_normalization(full_0, training = training)
    full_1 = tf.layers.conv2d_transpose(full_0_norm,5,5,strides = 1,padding = 'same',activation = tf.nn.elu,name = 'ballout')

    params_0 = tf.layers.dense(params,1*1*512,activation=None,kernel_initializer=tf.random_uniform_initializer(0.1))
    params_1 = tf.layers.conv2d_transpose(tf.reshape(params_0,shape =(-1,1,1,512)),256,5,strides = 2,padding = 'same',activation = tf.nn.elu)
    params_1_norm = tf.layers.batch_normalization(params_1, training=training)
    params_2 = tf.layers.conv2d_transpose(params_1_norm,128,1,strides = 2,padding = 'same',activation = tf.nn.elu)
    params_2_norm = tf.layers.batch_normalization(params_2, training=training)
    params_3 = tf.layers.conv2d_transpose(params_2_norm,64,2,strides = 2,padding = 'same',activation = tf.nn.elu)
    params_3_norm = tf.layers.batch_normalization(params_3, training=training)
    params_4 = tf.layers.conv2d_transpose(params_3_norm,32,3,strides = 2,padding = 'same',activation = tf.nn.elu)
    params_4_norm = tf.layers.batch_normalization(params_4, training=training)
    params_5 = tf.layers.conv2d_transpose(params_4_norm,16,4,strides = 2,padding = 'same',activation = tf.nn.elu)
    params_5_norm = tf.layers.batch_normalization(params_5, training=training)
    params_6 = tf.layers.conv2d_transpose(params_5_norm,5,5,strides = 2,padding = 'same',activation = tf.nn.elu,name = 'backout')

    image = tf.nn.sigmoid(tf.add(full_1,params_6),name = 'full_image')


    # (pt.wrap(together). ## Now the scalar input and the grid is merged
    #         deconv2d(5,16,stride = 2).
    #         deconv2d(5,5,stride = 2,activation_fn=tf.nn.sigmoid)).tensor # 32x32x4 ## Now ask to regenerate the meshgrid as well.
    return image

# def gener_model_dyn_fullsize_grid_color(grid,scalar,params):
#     '''This is a complicated architecture that will merge the grid and scalar outputs at an
#     appropriate output point.
#     '''
#     scalar_0 = tf.layers.dense(scalar,1*1*512,activation=None,kernel_initializer=tf.random_uniform_initializer(0.1))
#     scalar_1 = tf.layers.conv2d_transpose(tf.reshape(scalar_0,shape =(-1,1,1,512)),256,5,strides = 2,padding = 'same',activation = tf.nn.elu)
#     scalar_2 = tf.layers.conv2d_transpose(scalar_1,128,5,strides = 2,padding = 'same',activation = tf.nn.elu)
#     scalar_3 = tf.layers.conv2d_transpose(scalar_2,64,5,strides = 2,padding = 'same',activation = tf.nn.elu)
#     scalar_4 = tf.layers.conv2d_transpose(scalar_3,32,5,strides = 2,padding = 'same',activation = tf.nn.elu)
#     scalar_5 = tf.layers.conv2d_transpose(scalar_4,32,5,strides = 2,padding = 'same',activation = tf.nn.elu)
#     scalar_6 = tf.layers.conv2d_transpose(scalar_5,32,5,strides = 2,padding = 'same',activation = tf.nn.elu)
#
#     grid_0 = tf.layers.conv2d_transpose(grid,16,5,strides = 1,padding = 'same',activation = tf.nn.elu)
#     grid_1 = tf.layers.conv2d_transpose(grid_0,16,5,strides = 1,padding = 'same',activation = tf.nn.elu)
#
#     together = tf.concat((scalar_6,grid_1),axis = -1)
#
#     full_0 = tf.layers.conv2d_transpose(together,16,5,strides = 1,padding = 'same',activation = tf.nn.elu)
#     full_1 = tf.layers.conv2d_transpose(full_0,5,5,strides = 1,padding = 'same',activation = tf.nn.sigmoid,name = 'fullball')
#
#     back_0 = tf.layers.dense(params,1*1*512,activation=None,kernel_initializer=tf.random_uniform_initializer(0.1))
#     back_1 = tf.layers.conv2d_transpose(tf.reshape(back_0,shape =(-1,1,1,512)),256,5,strides = 2,padding = 'same',activation = tf.nn.elu)
#     back_2 = tf.layers.conv2d_transpose(back_1,128,5,strides = 2,padding = 'same',activation = tf.nn.elu)
#     back_3 = tf.layers.conv2d_transpose(back_2,64,5,strides = 2,padding = 'same',activation = tf.nn.elu)
#     back_4 = tf.layers.conv2d_transpose(back_3,32,5,strides = 2,padding = 'same',activation = tf.nn.elu)
#     back_5 = tf.layers.conv2d_transpose(back_4,32,5,strides = 2,padding = 'same',activation = tf.nn.elu)
#     back_6 = tf.layers.conv2d_transpose(back_5,32,5,strides = 2,padding = 'same',activation = tf.nn.elu,name = 'fullback')
#
#     full_withback = tf.add(full_1,back_6)
#
#     final = tf.layers.conv2d_transpose(full_withback,5,5,strides=2,padding = 'same',activation = tf.nn.sigmoid,name = 'final')
#     # (pt.wrap(together). ## Now the scalar input and the grid is merged
#     #         deconv2d(5,16,stride = 2).
#     #         deconv2d(5,5,stride = 2,activation_fn=tf.nn.sigmoid)).tensor # 32x32x4 ## Now ask to regenerate the meshgrid as well.
#     return final

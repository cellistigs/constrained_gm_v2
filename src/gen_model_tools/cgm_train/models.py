## Define the architectures used in the constrained generative model setting.
import sys
import numpy as np
import tensorflow as tf
import prettytensor as pt
from cgm_train.deconv import deconv2d
from config import imsize,dim_z,dim_v

# Global network parameters
def recog_model_regress_split(input_tensor,output_dim = dim_z+dim_v):
    ''' The input to this network (input_tensor) is a set of batches, each of which has
    temporal ordering. i.e. the examples in a single batch follow temporal dynamics
    We do this because we want to take the batched output, and apply dynamical transformations
    to it in order to generate samples from the appropriate approximate posterior.
    We assume that it comes as a 2d tensor, as [batch, everything]
    '''
    ## We assume we have a latent system with dimension 2

    ## Construct the shared layers of the network
    shared_layers = (pt.wrap(input_tensor).
                     reshape([None, imsize, imsize, 3]). ## Reshape input
                     conv2d(5, 16, stride=2). ## Three layers of convolution: 32x32x4
                     conv2d(5, 32, stride=2). ## 16x16x8
                     conv2d(5, 64, stride=2). ## 8x8x16
                     conv2d(5, 128, stride=2). ## 4x4x32
                     conv2d(5, 256, stride=2). ## 2x2x
                     conv2d(5, 512, stride=2). ## 1x1x
                     dropout(0.9).
                     flatten())
    # shared_layers = (pt.wrap(input_tensor).
    #                  reshape([None, dim_x, dim_x, 3]). ## Reshape input
    #                  conv2d(5, 16, stride=2). ## Three layers of convolution: 32x32x4
    #                  conv2d(5, 32, stride=2). ## 16x16x8
    #                  conv2d(5, 64, stride=2). ## 8x8x16
    #                  dropout(0.9).
    #                  flatten())

    dyn_output = shared_layers.fully_connected(output_dim,activation_fn=tf.nn.sigmoid,name = 'out',weights=tf.random_uniform_initializer(0.1)).tensor
    # stat_output = shared_layers.fully_connected(dim_v,activation_fn=tf.nn.sigmoid,name = 'stat',weights=tf.random_uniform_initializer(0.1)).tensor
    return dyn_output


def recog_model_regress(input_tensor):
    ''' The input to this network (input_tensor) is a set of batches, each of which has
    temporal ordering. i.e. the examples in a single batch follow temporal dynamics
    We do this because we want to take the batched output, and apply dynamical transformations
    to it in order to generate samples from the appropriate approximate posterior.
    We assume that it comes as a 2d tensor, as [batch, everything]
    '''
    ## We assume we have a latent system with dimension 2

    ## Construct the shared layers of the network
    shared_layers = (pt.wrap(input_tensor).
                     reshape([None, imsize, imsize, 3]). ## Reshape input
                     conv2d(5, 16, stride=2). ## Three layers of convolution: 32x32x4
                     conv2d(5, 32, stride=2). ## 16x16x8
                     conv2d(5, 64, stride=2). ## 8x8x16
                     conv2d(5, 128, stride=2). ## 4x4x32
                     conv2d(5, 256, stride=2). ## 2x2x
                     conv2d(5, 512, stride=2). ## 1x1x
                     dropout(0.9).
                     flatten())
    # shared_layers = (pt.wrap(input_tensor).
    #                  reshape([None, dim_x, dim_x, 3]). ## Reshape input
    #                  conv2d(5, 16, stride=2). ## Three layers of convolution: 32x32x4
    #                  conv2d(5, 32, stride=2). ## 16x16x8
    #                  conv2d(5, 64, stride=2). ## 8x8x16
    #                  dropout(0.9).
    #                  flatten())

    dyn_output = shared_layers.fully_connected(dim_z+dim_v,activation_fn=tf.nn.sigmoid,name = 'out',weights=tf.random_uniform_initializer(0.1)).tensor
    # stat_output = shared_layers.fully_connected(dim_v,activation_fn=tf.nn.sigmoid,name = 'stat',weights=tf.random_uniform_initializer(0.1)).tensor
    return dyn_output

## The only way in which this model differs from the one above is that the original
## image is two filters deeper, accounting for a meshgrid of x and y positions.
def recog_model_regress_motion(input_tensor,dim_z,dim_x,dim_v):
    ''' The input to this network (input_tensor) is a set of batches, each of which has
    temporal ordering. i.e. the examples in a single batch follow temporal dynamics
    We do this because we want to take the batched output, and apply dynamical transformations
    to it in order to generate samples from the appropriate approximate posterior.
    We assume that it comes as a 2d tensor, as [batch, everything]
    '''
    ## We assume we have a latent system with dimension 2

    ## Construct the shared layers of the network
    shared_layers = (pt.wrap(input_tensor).
                     reshape([None, dim_x, dim_x, 5]). ## Reshape input
                     conv2d(5, 16, stride=2). ## Three layers of convolution: 32x32x4
                     conv2d(5, 32, stride=2). ## 16x16x8
                     conv2d(5, 64, stride=2). ## 8x8x16
                     conv2d(5, 128, stride=2). ## 4x4x32
                     conv2d(5, 256, stride=2). ## 2x2x
                     conv2d(5, 512, stride=2). ## 1x1x
                     dropout(0.9).
                     flatten())
    # shared_layers = (pt.wrap(input_tensor).
    #                  reshape([None, dim_x, dim_x, 3]). ## Reshape input
    #                  conv2d(5, 16, stride=2). ## Three layers of convolution: 32x32x4
    #                  conv2d(5, 32, stride=2). ## 16x16x8
    #                  conv2d(5, 64, stride=2). ## 8x8x16
    #                  dropout(0.9).
    #                  flatten())

    dyn_output = shared_layers.fully_connected(dim_z,activation_fn=tf.nn.sigmoid,name = 'out',weights=tf.random_uniform_initializer(0.1)).tensor
    # stat_output = shared_layers.fully_connected(dim_v,activation_fn=tf.nn.sigmoid,name = 'stat',weights=tf.random_uniform_initializer(0.1)).tensor
    return dyn_output

## This network applies the VIN Visual encoder architecture to a single image.
def recog_model_regress_motion_full(input_tensor,dim_z,dim_x,dim_v):
    ''' The input to this network (input_tensor) is a set of batches, each of which has
    temporal ordering. i.e. the examples in a single batch follow temporal dynamics
    We do this because we want to take the batched output, and apply dynamical transformations
    to it in order to generate samples from the appropriate approximate posterior.
    We assume that it comes as a 2d tensor, as [batch, everything]
    '''
    ##
    batch_size = 50
    ## Pre-specify meshgrid position info;
    index = np.linspace(0,1,dim_x)

    xx,yy = np.meshgrid(index,index)

    # posgrid = tf.cast(tf.constant(np.concatenate((xx.reshape(imsize,imsize,1),yy.reshape(imsize,imsize,1)),axis = 2)),tf.float32)
    posgrid = tf.cast(tf.constant(np.repeat(np.concatenate((xx.reshape(1,dim_x,dim_x,1),yy.reshape(1,dim_x,dim_x,1)),axis = 3),batch_size,axis = 0)),tf.float32)
    # image = tf.concat((tf.cast(image,tf.float32),posgrid),axis = 2)

    ## Apply two independent convolutions with different filter sizes
    big_preprocess = (pt.wrap(input_tensor).
                      reshape([None,dim_x,dim_x,3]).
                      conv2d(10,4,batch_normalize=True).
                      conv2d(10,4,batch_normalize=True))

    med_preprocess = (pt.wrap(input_tensor).
                      reshape([None,dim_x,dim_x,3]).
                      conv2d(6,8,batch_normalize=True).
                      conv2d(6,8,batch_normalize=True))

    all_preprocess = (pt.wrap(input_tensor).
                      reshape([None,dim_x,dim_x,3]).
                      conv2d(3,16,batch_normalize=True).
                      conv2d(3,16,batch_normalize=True). ## Small preprocessing
                      concat(-1, other_tensors = (big_preprocess,med_preprocess)).
                      conv2d(3,16,batch_normalize=True).
                      conv2d(3,16,batch_normalize=True)).tensor

    ## Now concatenate spatial information, giving a (image x image x 19 dimensional image):
    with_spatial = tf.concat((all_preprocess,posgrid),axis = -1)


    ## Construct the shared layers of the network
    shared_layers = (pt.wrap(with_spatial).
                     reshape([None, dim_x, dim_x, 18]). ## Reshape input
                     conv2d(3, 16, stride=2,batch_normalize=True). ## Three layers of convolution: 32x32x4
                     conv2d(3, 16, stride=2,batch_normalize=True). ## 16x16x8
                     conv2d(3, 16, stride=2,batch_normalize=True). ## 8x8x16
                     conv2d(3, 32, stride=2,batch_normalize=True). ## 4x4x32
                     conv2d(3, 32, stride=2,batch_normalize=True). ## 2x2x
                     conv2d(3, 32, stride=2,batch_normalize=True). ## 1x1x
                     dropout(0.9).
                     flatten())
    # shared_layers = (pt.wrap(input_tensor).
    #                  reshape([None, dim_x, dim_x, 3]). ## Reshape input
    #                  conv2d(5, 16, stride=2). ## Three layers of convolution: 32x32x4
    #                  conv2d(5, 32, stride=2). ## 16x16x8
    #                  conv2d(5, 64, stride=2). ## 8x8x16
    #                  dropout(0.9).
    #                  flatten())

    dyn_output = shared_layers.fully_connected(dim_z,activation_fn=tf.nn.sigmoid,name = 'out',weights=tf.random_uniform_initializer(0.1)).tensor
    # stat_output = shared_layers.fully_connected(dim_v,activation_fn=tf.nn.sigmoid,name = 'stat',weights=tf.random_uniform_initializer(0.1)).tensor
    return dyn_output

def recog_model_dyn(input_tensor,dim_z,dim_x):
    ''' The input to this network (input_tensor) is a set of batches, each of which has
    temporal ordering. i.e. the examples in a single batch follow temporal dynamics
    We do this because we want to take the batched output, and apply dynamical transformations
    to it in order to generate samples from the appropriate approximate posterior.
    We assume that it comes as a 2d tensor, as [batch, everything]
    '''
    ## We assume we have a latent system with dimension 2
    dim_v = 2

    ## Construct the shared layers of the network
    shared_layers = (pt.wrap(input_tensor).
                     reshape([None, imsize, imsize, 3]). ## Reshape input
                     conv2d(5, 16, stride=2). ## Three layers of convolution: 32x32x4
                     conv2d(5, 32, stride=2). ## 16x16x8
                     conv2d(5, 64, stride=2). ## 8x8x16
                     conv2d(5, 128, stride=2). ## 4x4x32
                     conv2d(5, 256, stride=2). ## 2x2x
                     dropout(0.9).
                     flatten())
    # shared_layers = (pt.wrap(input_tensor).
    #                  reshape([None, dim_x, dim_x, 3]). ## Reshape input
    #                  conv2d(5, 16, stride=2). ## Three layers of convolution: 32x32x4
    #                  conv2d(5, 32, stride=2). ## 16x16x8
    #                  conv2d(5, 64, stride=2). ## 8x8x16
    #                  dropout(0.9).
    #                  flatten())

    mean_output = shared_layers.fully_connected(dim_z,activation_fn=None,name = 'means',weights=tf.random_uniform_initializer(0.1)).tensor
    covar_output = shared_layers.fully_connected(dim_z*dim_z,activation_fn=tf.nn.tanh,name = 'covars',weights=tf.random_uniform_initializer(0.1)).tensor
    return mean_output,covar_output

def recog_model_dyn_stat(input_tensor,dim_z,dim_x,dim_v,batch_size):
    ''' The input to this network (input_tensor) is a set of batches, each of which has
    temporal ordering. i.e. the examples in a single batch follow temporal dynamics
    We do this because we want to take the batched output, and apply dynamical transformations
    to it in order to generate samples from the appropriate approximate posterior.
    We assume that it comes as a 2d tensor, as [batch, everything]
    '''
    ## We assume we have a latent system with dimension 2

    ## Construct the shared layers of the network
    shared_layers = (pt.wrap(input_tensor).
                     reshape([None, dim_x, dim_x, 3]). ## Reshape input
                     conv2d(5, 16, stride=2). ## Three layers of convolution: 32x32x4
                     conv2d(5, 32, stride=2). ## 16x16x8
                     conv2d(5, 64, stride=2). ## 8x8x16
                     conv2d(5, 128, stride=2). ## 4x4x32
                     conv2d(5, 256, stride=2). ## 2x2x
                     dropout(0.9).
                     flatten())
    # shared_layers = (pt.wrap(input_tensor).
    #                  reshape([None, dim_x, dim_x, 3]). ## Reshape input
    #                  conv2d(5, 16, stride=2). ## Three layers of convolution: 32x32x4
    #                  conv2d(5, 32, stride=2). ## 16x16x8
    #                  conv2d(5, 64, stride=2). ## 8x8x16
    #                  dropout(0.9).
    #                  flatten())

    mean_output = shared_layers.fully_connected(dim_z,activation_fn=None,name = 'means',weights=tf.random_uniform_initializer(0.1)).tensor
    covar_output = shared_layers.fully_connected(dim_z*dim_z,activation_fn=tf.nn.tanh,name = 'covars',weights=tf.random_uniform_initializer(0.1)).tensor
    # We do something outside of the prettytensor library to reshape into a way that combines across batches.
    # First simplify to someting of dimension z in batches:
    stat_batchwise_mean = shared_layers.fully_connected(dim_v,activation_fn=None,weights=tf.random_uniform_initializer(0.1)).tensor
    stat_batchwise_var = shared_layers.fully_connected(dim_v,activation_fn=None,weights=tf.random_uniform_initializer(0.1)).tensor
    # Now multiply across averages: essentilly as a weighted sum (initialize as sum):
    stat_weights_mean = tf.get_variable('Stat_weights_mean',shape = [1,batch_size],dtype = tf.float32,initializer = tf.constant_initializer(1.0/batch_size))
    stat_weights_var = tf.get_variable('Stat_weights_var',shape = [1,batch_size],dtype = tf.float32,initializer = tf.constant_initializer(1.0/batch_size))
    stat_avg_mean = tf.matmul(stat_weights_mean,stat_batchwise_mean)
    stat_avg_var = tf.matmul(stat_weights_var,stat_batchwise_var)

    return mean_output,covar_output,stat_avg_mean,stat_avg_var

def recog_model_dyn_stat_full(input_tensor,dim_z,dim_x,dim_v,batch_size):
    ''' The input to this network (input_tensor) is a set of batches, each of which has
    temporal ordering. i.e. the examples in a single batch follow temporal dynamics
    We do this because we want to take the batched output, and apply dynamical transformations
    to it in order to generate samples from the appropriate approximate posterior.
    We assume that it comes as a 2d tensor, as [batch, everything]
    '''
    ## We assume we have a latent system with dimension 2

    ## Construct the shared layers of the network
    shared_layers = (pt.wrap(input_tensor).
                     reshape([None, imsize, imsize, 3]). ## Reshape input
                     conv2d(5, 16, stride=2). ## Three layers of convolution: 32x32x4
                     conv2d(5, 32, stride=2). ## 16x16x8
                     conv2d(5, 64, stride=2). ## 8x8x16
                     conv2d(5, 128, stride=2). ## 4x4x32
                     conv2d(5, 256, stride=2). ## 2x2x
                     conv2d(5, 512, stride=2). ## 2x2x
                     dropout(0.9).
                     flatten())
    # shared_layers = (pt.wrap(input_tensor).
    #                  reshape([None, dim_x, dim_x, 3]). ## Reshape input
    #                  conv2d(5, 16, stride=2). ## Three layers of convolution: 32x32x4
    #                  conv2d(5, 32, stride=2). ## 16x16x8
    #                  conv2d(5, 64, stride=2). ## 8x8x16
    #                  dropout(0.9).
    #                  flatten())

    mean_output = shared_layers.fully_connected(dim_z,activation_fn=None,name = 'means',weights=tf.random_uniform_initializer(0.1)).tensor
    covar_output = shared_layers.fully_connected(dim_z*dim_z,activation_fn=tf.nn.tanh,name = 'covars',weights=tf.random_uniform_initializer(0.1)).tensor
    # We do something outside of the prettytensor library to reshape into a way that combines across batches.
    # First simplify to someting of dimension z in batches:
    stat_batchwise_mean = shared_layers.fully_connected(dim_v,activation_fn=None,weights=tf.random_uniform_initializer(0.1)).tensor
    stat_batchwise_var = shared_layers.fully_connected(dim_v,activation_fn=None,weights=tf.random_uniform_initializer(0.1)).tensor
    # Now multiply across averages: essentilly as a weighted sum (initialize as sum):
    stat_weights_mean = tf.get_variable('Stat_weights_mean',shape = [1,batch_size],dtype = tf.float32,initializer = tf.constant_initializer(1.0/batch_size))
    stat_weights_var = tf.get_variable('Stat_weights_var',shape = [1,batch_size],dtype = tf.float32,initializer = tf.constant_initializer(1.0/batch_size))
    stat_avg_mean = tf.matmul(stat_weights_mean,stat_batchwise_mean)
    stat_avg_var = tf.matmul(stat_weights_var,stat_batchwise_var)

    return mean_output,covar_output,stat_avg_mean,stat_avg_var


def gener_model_dyn(hidden_activations):
    '''The input to this network (hidden_activations) is a set of sampled activations that
    represents hidden activations across a batch. They are correlated by means of the structure imposed
    on the noise that they experience when they are sampled together, but here they should be shaped in a
    batch-friendly setup, that is once again a 2d tensor: [batch,everything]. We will first connect to a
    fully connected layer in order to generate input that is correctly shaped for deconvolution that mirrors
    the recognition model.
    '''
    return (pt.wrap(hidden_activations).
            fully_connected(2*2*256,activation_fn=None,weights=tf.random_uniform_initializer(0.1)).
            reshape([None,2,2,256]).
            deconv2d(5,128,stride = 2).
            deconv2d(5,64,stride = 2).
            deconv2d(5,32,stride = 2).
            deconv2d(5,16,stride = 2).
            deconv2d(5,3,stride = 2,activation_fn=tf.nn.sigmoid)).tensor # 32x32x4

def gener_model_dyn_full(hidden_activations):
    '''The input to this network (hidden_activations) is a set of sampled activations that
    represents hidden activations across a batch. They are correlated by means of the structure imposed
    on the noise that they experience when they are sampled together, but here they should be shaped in a
    batch-friendly setup, that is once again a 2d tensor: [batch,everything]. We will first connect to a
    fully connected layer in order to generate input that is correctly shaped for deconvolution that mirrors
    the recognition model.
    '''
    return (pt.wrap(hidden_activations).
            fully_connected(1*1*512,activation_fn=None,weights=tf.random_uniform_initializer(0.1)).
            reshape([None,1,1,512]).
            deconv2d(5,256,stride = 2).
            deconv2d(5,128,stride = 2).
            deconv2d(5,64,stride = 2).
            deconv2d(5,32,stride = 2).
            deconv2d(5,16,stride = 2).
            deconv2d(5,3,stride = 2,activation_fn=tf.nn.sigmoid)).tensor # 32x32x4

def gener_model_dyn_full_motion(hidden_activations):
    '''The input to this network (hidden_activations) is a set of sampled activations that
    represents hidden activations across a batch. They are correlated by means of the structure imposed
    on the noise that they experience when they are sampled together, but here they should be shaped in a
    batch-friendly setup, that is once again a 2d tensor: [batch,everything]. We will first connect to a
    fully connected layer in order to generate input that is correctly shaped for deconvolution that mirrors
    the recognition model.
    '''
    return (pt.wrap(hidden_activations).
            fully_connected(1*1*512,activation_fn=None,weights=tf.random_uniform_initializer(0.1)).
            reshape([None,1,1,512]).
            deconv2d(5,256,stride = 2).
            deconv2d(5,128,stride = 2).
            deconv2d(5,64,stride = 2).
            deconv2d(5,32,stride = 2).
            deconv2d(5,16,stride = 2).
            deconv2d(5,5,stride = 2,activation_fn=tf.nn.sigmoid)).tensor # 32x32x4 ## Now ask to regenerate the meshgrid as well.

def gener_model_dyn_full_grid_basic(grid,scalar):
    '''This is a complicated architecture that will merge the grid and scalar outputs at an
    appropriate output point.
    '''
    scalar_0 = tf.layers.dense(scalar,1*1*512,activation=None,weights=tf.random_uniform_initializer(0.1))
    scalar_1 = tf.layers.conv2d_transpose(tf.reshape(scalar_0,[None,1,1,512]),256,5,strides = 2,padding = 'same',activation = tf.nn.elu)
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
    return full_1


def gener_model_dyn_full_grid(grid,scalar):
    '''This is a complicated architecture that will merge the grid and scalar outputs at an
    appropriate output point.
    '''
    scalar_only = (pt.wrap(scalar).
                   fully_connected(1*1*512,activation_fn=None,weights=tf.random_uniform_initializer(0.1)).
                   reshape([None,1,1,512]).
                   deconv2d(5,256,stride = 2).
                   deconv2d(5,128,stride = 2).
                   deconv2d(5,64,stride = 2).
                   deconv2d(5,32,stride = 2)).tensor ## 16x16x32

    grid_only = (pt.wrap(grid).
            deconv2d(5,16,stride = 1).
            deconv2d(5,16,stride = 1)).tensor

    together = tf.concat((scalar_only,grid_only),axis = -1)

    full = (pt.wrap(together). ## Now the scalar input and the grid is merged
            deconv2d(5,16,stride = 2).
            deconv2d(5,5,stride = 2,activation_fn=tf.nn.sigmoid)).tensor # 32x32x4 ## Now ask to regenerate the meshgrid as well.
    return full

def gener_model_mini(hidden_activations):
    '''The input to this network (hidden_activations) is a set of sampled activations that
    represents hidden activations across a batch. They are correlated by means of the structure imposed
    on the noise that they experience when they are sampled together, but here they should be shaped in a
    batch-friendly setup, that is once again a 2d tensor: [batch,everything]. We will first connect to a
    fully connected layer in order to generate input that is correctly shaped for deconvolution that mirrors
    the recognition model.
    '''
    return (pt.wrap(hidden_activations).
            fully_connected(10,activation_fn=tf.nn.sigmoid,weights=tf.random_uniform_initializer(0.1)).
            fully_connected(1,activation_fn=tf.nn.sigmoid,weights=tf.random_uniform_initializer(0.1))).tensor # 32x32x4

    # return (pt.wrap(hidden_activations).
    #         fully_connected(8*8*64,activation_fn=None,weights=tf.random_uniform_initializer(0.1)).
    #         reshape([None,8,8,64]).
    #         deconv2d(5,32,stride = 2).
    #         deconv2d(5,16,stride = 2).
    #         deconv2d(5,3,stride = 2,activation_fn=tf.nn.sigmoid)).tensor # 32x32x4

# def gener_model_dyn(hidden_activations):
#     '''The input to this network (hidden_activations) is a set of sampled activations that
#     represents hidden activations across a batch. They are correlated by means of the structure imposed
#     on the noise that they experience when they are sampled together, but here they should be shaped in a
#     batch-friendly setup, that is once again a 2d tensor: [batch,everything]. We will first connect to a
#     fully connected layer in order to generate input that is correctly shaped for deconvolution that mirrors
#     the recognition model.
#     '''
#     return (pt.wrap(hidden_activations).
#             fully_connected(2*2*256,activation_fn=None,weights=tf.random_uniform_initializer(0.1)).
#             reshape([None,2,2,256]).
#             deconv2d(5,128,stride = 2).
#             deconv2d(5,64,stride = 2).
#             deconv2d(5,32,stride = 2).
#             deconv2d(5,16,stride = 2).
#             deconv2d(5,3,stride = 2,activation_fn=tf.nn.sigmoid)).tensor # 32x32x4

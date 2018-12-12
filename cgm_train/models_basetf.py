import sys
import numpy as np
import tensorflow as tf
from config import batch_size,imsize,dim_z,dim_v,training,nb_channels


### We want a vanilla convolutional autoencoder as a performance baseline. Give the
### convolution and convolution transpose parts:

# Define a convolve+batchnorm layer. Both of these assume that the input has been appropriately shaped
# prior to entry.
# Name gives a prefix for the names of each layer included here- like a miniature scope.
def conv2d_bn(input,filters,kernel,name,
              strides = 2,
              padding = 'same',
              activation = tf.nn.elu,
              training = False,
              bn = True):
    if bn == True:
        convd = tf.layers.conv2d(input,filters,kernel,strides = strides,padding = padding,activation = activation,name = name+'/conv')
        normed = tf.layers.batch_normalization(convd,training = training, name = name+'/bn')
    else:
        normed = tf.layers.conv2d(input,filters,kernel,strides = strides,padding = padding,activation = activation,name = name+'/conv')
    return normed

# Likewise define an adjoint convolution:
def adjconv2d_bn(input,filters,kernel,name,
                 strides = 2,
                 padding = 'same',
                 activation = tf.nn.elu,
                 training = False,
                 bn = True):
    if bn == True:
        adjconvd = tf.layers.conv2d_transpose(input,filters,kernel,strides = strides,padding = padding,activation = activation,name = name+'/adjconv')
        normed = tf.layers.batch_normalization(adjconvd,training = training, name = name+'/bn')
    else:
        normed = tf.layers.conv2d_transpose(input,filters,kernel,strides = strides,padding = padding,activation = activation,name = name+'/adjconv')
    return normed

# Define a strided layer recursion on convolutions using the above:
def conv2d_bn_to_vector(input,name,
                        kernel=5,
                        seed_filter_nb = 4,
                        strides = 2,
                        filter_seq = None,
                        training=True):
    # define the vector of filters to use if not specified. Default to convolution down to 1x1.
    nb_layers = np.ceil(np.log2(imsize)/np.log2(strides))
    if filter_seq == None:
        nb_layers = np.ceil(np.log2(imsize)/np.log2(strides))
        filter_seq = [seed_filter_nb*(2**layer_index) for layer_index in range(int(nb_layers))]
    else:
        try:
            len(filter_seq) <= nb_layers
        except ValueError:
            print('Filter sequence suggests impossible architecture for input and strides given.')

    # Iterate through the architecture:
    x = input
    for i,filter_nb in enumerate(filter_seq):
        # Last layer gets no batch norm. This may not matter for convolution down.
        if i == len(filter_seq)-1:
            x = conv2d_bn(x,filter_nb,kernel,name = name+'/layer'+str(i),bn = False)
        else:
            x = conv2d_bn(x,filter_nb,kernel,name = name+'/layer'+str(i),training = training)
    return x

# Define a strided layer recursion on adjoint convolutions using the above:
def adjconv2d_bn_to_vector(input,name,
                           kernel=5,
                           seed_filter_nb = 4,
                           strides = 2,
                           filter_seq = None,
                           training = True):
    # define the vector of filters to use if not specified. Default to convolution from to 1x1 to image.
    nb_layers = np.ceil(np.log2(imsize)/np.log2(strides))
    if filter_seq == None:
        nb_layers = np.ceil(np.log2(imsize)/np.log2(strides))
        filter_seq = [seed_filter_nb*(2**layer_index) for layer_index in range(int(nb_layers))]
        filter_seq = filter_seq[::-1]
        # Replace the last layer with image-depth filters:
        filter_seq[-1] = nb_channels
    else:
        try:
            len(filter_seq) <= nb_layers
        except ValueError:
            print('Filter sequence suggests impossible architecture for input and strides given.')
        try:
            filter_seq[-1] = nb_channels

        except ValueError:
            print('Output of last layer should have depth matching image.')
    # Iterate through the architecture:
    x = input
    for i,filter_nb in enumerate(filter_seq):

        if i == len(filter_seq)-1:
            x = adjconv2d_bn(x,filter_nb,kernel,name = name+'/layer'+str(i),activation = tf.nn.sigmoid,bn = False)
        else:
            x = adjconv2d_bn(x,filter_nb,kernel,name = name+'/layer'+str(i),training=training)
    return x

## Vanilla recognition model. Infers the means and standard deviation for our factor
## model prior.
## q(z|x)
def recog_model_vanilla(input_tensor,dim_z,name,strides = 2,seed_filter_nb = 4,training=True):
    "A vanilla architecture to process inputs into latent variable parameters."
    nb_layers = np.ceil(np.log2(imsize)/np.log2(strides))
    filter_seq = [seed_filter_nb*(2**layer_index) for layer_index in range(int(nb_layers))]
    input_shaped = tf.reshape(input_tensor,[-1,imsize,imsize,nb_channels])
    conv_out = conv2d_bn_to_vector(input_shaped,strides = strides,filter_seq=filter_seq,name = name,training=training)
    conv_out_reshaped = tf.reshape(conv_out,[-1,filter_seq[-1]])
    inference_means = tf.layers.dense(conv_out_reshaped,dim_z,activation = None,name = name+'/means')
    inference_logstds = tf.layers.dense(conv_out_reshaped,dim_z,activation = None,name = name+'/logstds')
    return inference_means,inference_logstds

## Vanilla generative model. Infers the mean of the image from a sample of the latent variables.
## p(x|z)
def gener_model_vanilla(input_tensor,name,strides = 2,seed_filter_nb = 4,training=True):
    "A vanilla architecture to generate data from samples of the generative model."

    nb_layers = np.ceil(np.log2(imsize)/np.log2(strides))
    filter_seq = [seed_filter_nb*(2**layer_index) for layer_index in range(int(nb_layers))]
    filter_seq = filter_seq[::-1]

    input_proj = tf.layers.dense(input_tensor,filter_seq[0],activation = None)
    input_shaped = tf.reshape(input_proj,[-1,1,1,filter_seq[0]])
    image = adjconv2d_bn_to_vector(input_shaped,strides=strides,filter_seq= filter_seq,name = name,training=training)

    ## Already in 0-1 range. Should we add some noise?
    return image

## Category recognition model. Infers the probability of each category from data.
## q(y|x).
def recog_model_cat(input_tensor,dim_y,name,strides = 2,seed_filter_nb = 4,training=True):
    "A vanilla architecture to process inputs into latent variable parameters."
    nb_layers = np.ceil(np.log2(imsize)/np.log2(strides))
    filter_seq = [seed_filter_nb*(2**layer_index) for layer_index in range(int(nb_layers))]
    input_shaped = tf.reshape(input_tensor,[batch_size,imsize,imsize,nb_channels])
    conv_out = conv2d_bn_to_vector(input_shaped,strides = strides,filter_seq=filter_seq,name = name,training=training)
    conv_out_reshaped = tf.reshape(conv_out,[batch_size,-1])
    cat_probs = tf.layers.dense(conv_out_reshaped,dim_y,activation = tf.nn.softmax,name = name+'/catprobs')

    return cat_probs

## Generative model to infer distribution of continuous latents from category labels, p(z|y)
def gener_model_mixture(input_tensor,dim_z,name):
    # Use a two layer mlp:
    nb_hidden = 512
    # First layer should take batch x 1:
    hidden = tf.layers.dense(input_tensor,nb_hidden,activation = tf.nn.elu,name = name+'/gmix1')
    # Second layer should return dimension z
    mixmeans = tf.layers.dense(hidden,dim_z,activation = None,name = name+'/gmix_means')
    mixlogstds = tf.layers.dense(hidden,dim_z,activation = None, name = name+'/gmix_logstds')

    return mixmeans,mixlogstds

## Part of recognition model to infer lower-d representation of images that can be loaded into
## by the weights from the vanilla network:
def recog_model_imageprocess(input_tensor,dim_z,name,strides = 2,seed_filter_nb = 4,training=True):
    "A vanilla architecture to process inputs into latent variable parameters."
    nb_layers = np.ceil(np.log2(imsize)/np.log2(strides))
    filter_seq = [seed_filter_nb*(2**layer_index) for layer_index in range(int(nb_layers))]
    input_shaped = tf.reshape(input_tensor,[batch_size,imsize,imsize,nb_channels])
    conv_out = conv2d_bn_to_vector(input_shaped,strides = strides,filter_seq=filter_seq,name = name,training=training)
    conv_out_reshaped = tf.reshape(conv_out,[batch_size,-1])
    return conv_out_reshaped

## Recognition model to infer distribution of continuous latents from categories and pre-compressed
## latent codes q(z|x,y)
def recog_model_mixture(input_tensor,dim_y,dim_z,name):
    # First make a one-hot representation of possible latent codes:
    ## batch_size x cluster number
    index = [i for i in range(dim_y) for j in range(batch_size)]
    cat_labels = tf.one_hot(index,depth = dim_y,axis = -1)

    # Now we will repeat the dataset cluster number of times:
    rep_data = tf.tile(input_tensor,[dim_y,1])

    # Concatenate:
    full_data = tf.concat((cat_labels,rep_data),axis = 1)

    # MLP with two hidden layer:
    shared_hidden0 = tf.layers.dense(full_data,512,activation = tf.nn.elu,name = name+'/rmix_0')
    shared_hidden1 = tf.layers.dense(full_data,512,activation = tf.nn.elu,name = name+'/rmix_1')

    # Output into means and logstds:
    inference_means = tf.layers.dense(shared_hidden1,dim_z,activation = None,name = name+'recog_means')
    inference_logstds = tf.layers.dense(shared_hidden1,dim_z,activation = None, name = name+'recog_logstds')

    return inference_means,inference_logstds

## This is a construction that wraps the recognition and generative models, and
## handles the complexities involved in sampling multiple times from the prior.
## Also sets up the tensorboard summary statistics.
## TODO: Set up placeholder variables, and a switch for training vs. inference mode.
def VAE_vanilla_graph(input_tensor,dim_z,name,nb_samples = 5,training=True,sample_mean = False):
    mean,logstd = recog_model_vanilla(input_tensor,dim_z,name+'/recog',training=training)
    ## reparametrization trick
    # First broadcast mean and standard deviation appropriately
    mean_broadcast = tf.tile(tf.expand_dims(mean,0),(nb_samples,1,1))
    std_broadcast = tf.tile(tf.expand_dims(tf.exp(logstd),0),(nb_samples,1,1))

    # Sample noise:
    eps = tf.random_normal((nb_samples,batch_size,dim_z))

    # The trick itself:
    if sample_mean == True:
        samples = mean_broadcast+std_broadcast*eps
    else:
        samples = mean_broadcast

    # We use the map function to apply the transformation everything simultaneously:
    #gen_func = lambda input: gener_model_vanilla(input,name+'/gener',training=training)
    #out = tf.map_fn(gen_func,samples) ## Parallellize here?
    ## using map_fn does not play well with batch norm.
    samples_reshape = tf.reshape(samples,(nb_samples*batch_size,dim_z))
    out = gener_model_vanilla(samples,name+'/gener',training = training)
    out_reshaped = tf.reshape(out,(nb_samples,batch_size,imsize,imsize,nb_channels))

    ## In order to evaluate performance, we need to evaluate quantities that are related to
    ## the statistics of our variational distributions (parameters of z) and the final likelihood (samples of x)
    return out_reshaped,mean,logstd

## We extend the standard VAE construction to a Gaussian mixture VAE. This involves alterations to the


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

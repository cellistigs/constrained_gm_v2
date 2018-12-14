## A place to store all the costs that we implement:
import tensorflow as tf
import numpy as np
from config import batch_size,dim_z,imsize,dim_y

# Formulate the cost for a vanilla variational autoencoder:
def VAE_likelihood_MC(input,data_sample):
    nb_samples = tf.shape(data_sample)[0]
    data_samples_flat = tf.reshape(data_sample,[nb_samples,batch_size,-1])
    input_expand = tf.tile(tf.expand_dims(input,0),(nb_samples,1,1,1,1))
    input_flat = tf.reshape(input_expand,[nb_samples,batch_size,-1])
    se = -0.5*tf.reduce_sum(tf.square(input_flat-data_samples_flat),-1)
    rmse = tf.reduce_mean(se,0)
    # prefactor = -0.5*(imsize*imsize*3)*np.log(2*np.pi)*batch_size
    ## assume sigma 1 => log denominator is 0
    cost = tf.reduce_sum(rmse)
    # e = input_flat-data_sample
    # se = -0.5*tf.square(e)
    # mse = tf.reduce_mean(se,axis = 0) ## multiple samples
    # cost = tf.reduce_sum(mse) ## sum over the image and over the batch.
    return cost

#### Dimensions need to be figured out for all costs.
def GMVAE_likelihood_MC(input,data_sample,cat_probs_batch):
    print(input,data_sample,'shapes')
    # We need to account for tiling of the examples per
    nb_samples = tf.shape(data_sample)[0]
    data_samples_flat = tf.reshape(data_sample,[nb_samples,dim_y*batch_size,-1])
    input_expand = tf.tile(tf.expand_dims(input,0),(nb_samples,dim_y,1,1,1))
    input_flat = tf.reshape(input_expand,[nb_samples,dim_y*batch_size,-1])
    se = -0.5*tf.reduce_sum(tf.square(input_flat-data_samples_flat),-1)
    rmse = tf.reduce_mean(se,0)
    ## This is of dimension (dim_y*batch_size,)

    ## We will weight by the category vector
    weighted_rmse = tf.multiply(cat_probs_batch,rmse)

    # prefactor = -0.5*(imsize*imsize*3)*np.log(2*np.pi)*batch_size
    ## assume sigma 1 => log denominator is 0
    cost = tf.reduce_sum(rmse)
    # e = input_flat-data_sample
    # se = -0.5*tf.square(e)
    # mse = tf.reduce_mean(se,axis = 0) ## multiple samples
    # cost = tf.reduce_sum(mse) ## sum over the image and over the batch.
    return cost

def GMVAE_cluster_cost(samples,gener_means,gener_logstd,cat_probs_batch):
    # samples are of size (nb_samples,batch_size*dim_y,dim_z)
    # gener_means, gener_vars are of size (batch_size*dim_y,dim_z)
    # First we will reshape the means and variances to afford the right comparison:
    nb_samples = tf.shape(samples)[0]
    mean_broad = tf.tile(tf.expand_dims(gener_means,0),(nb_samples,1,1))
    log_std_broad = tf.tile(tf.expand_dims(gener_logstd,0),(nb_samples,1,1))

    ## Calculate the log likelihood of each sample: log[1/sqrt(2*pi*sig^2)exp(-0.5(s-mu)^2/sig^2)]
    pre = -0.5*np.log(2*np.pi)-log_std_broad
    lik = -0.5*tf.square(samples-mean_broad)/tf.square(tf.exp(log_std_broad))

    cost_unweighted = tf.reduce_sum(pre+lik,-1)

    cost_averaged = tf.reduce_mean(cost_unweighted,0)

    cost_weighted= tf.reduce_sum(tf.multiply(cat_probs_batch,cost_averaged))

    return cost_weighted

def GMVAE_prior_cost(cat_probs_batch):
    ## This is basically a constant factor to encourage uniformity:
    factor = tf.reshape(tf.log(1./dim_y),(1,1))
    factor_extended = tf.tile(factor,(batch_size*dim_y,1))
    cost = tf.reduce_sum(factor_extended)
    return cost

def GMVAE_normal_entropy(infer_log_stds,cat_probs_batch):
    ## Calculate the gaussian entropy of the inferred normal distributions:
    ## This is of shape (batch_size*dim_y,dim_z)
    elementwise = infer_log_stds+0.5*np.log(2*np.pi*np.exp(1))

    examplewise = tf.reduce_sum(elementwise,1)

    cost = tf.multiply(cat_probs_batch,examplewise)
    return tf.reduce_sum(cost)

def GMVAE_cat_entropy(cat_probs):
    datawise = -tf.reduce_sum(tf.multiply(cat_probs,tf.log(cat_probs)),1)
    return tf.reduce_sum(datawise)

def VAE_likelihood_MC_debug(input,data_sample,sigma):
    nb_samples = tf.shape(data_sample)[0]
    input_expand = tf.tile(tf.expand_dims(input,0),(nb_samples,1,1,1,1))
    e = input_expand-data_sample
    se = -0.5*tf.square(e)
    # mse = tf.reduce_mean(se,axis = 0) ## multiple samples
    # cost = tf.reduce_sum(mse) ## sum over the image and over the batch.
    return se

def D_kl_prior_cost(mean,log_sigma):
    per_data = dim_z/2.+0.5*(tf.reduce_sum(2*(log_sigma)-tf.square(mean)-tf.square(tf.exp(log_sigma)),axis =1))
    total = tf.reduce_sum(per_data)
    return total

###### Costs for fLDS

# The total cost that we will use to train our model is a
# sum of two parts (as is always true of the elbo)
# We can best parse our cost by considering it as an entropic term
# (that can be evaluated analytically) and an expectation over the
# log-joint distribution of x and z that we sample via SGVB.

# The entropy in Z can be formulated as a function of the
# Cholesky decomposition of Sigma (this is R).
def entropy_cost(R):
    ## Entropy for a gaussian is a simple function of the covariance.
    logdet = -2*tf.reduce_sum(tf.log(tf.diag_part(R)+1e-10))
    return logdet/2+dim_z*batch_size/2*np.log(2*np.pi*np.exp(1))

# The joint distribution can further be formulated as sum of a
# log likelihood and a prior over the structure of Z (which also encodes
# dynamical information)
def likelihood_cost(true_images,gen_images,gen_params):
    ## Let's impose a gaussian likelihood elementwise and call the
    ## log likelihood an RMSE:
    resvec_x = tf.reshape(true_images,[batch_size,-1])-tf.reshape(gen_images,[batch_size,-1])
    ## "Invert" your R_gen
    R_inv = (1./gen_params['R_gen']).astype('float32')
    R_inv_constant = R_inv[0]
    ##### NOTE: THIS SHOULD BE TRACE
    # rmse = -0.5*tf.reduce_sum(tf.matmul(resvec_x,tf.transpose(resvec_x)))*R_inv_constant
    rmse = -0.5*tf.trace(tf.matmul(resvec_x,tf.transpose(resvec_x)))*R_inv_constant
    ## If we were really multiplying elementwise by a matrix with R_inv on the
    ## diagonal, it would be different. ### NO IT WOULD NOT

    ## We also have a cost coming from the log determinant in the
    ## denominator:
    denom = 0.5*tf.reduce_sum(tf.log(R_inv))*batch_size
    prefactor = -0.5*(imsize*imsize*3)*np.log(2*np.pi)*batch_size
    return denom+rmse+prefactor

def regression_cost(true_images,gen_images):
    ## Let's impose a gaussian likelihood elementwise and call the
    ## log likelihood an RMSE:
    resvec_x = tf.reshape(true_images,[batch_size,-1])-tf.reshape(gen_images,[batch_size,-1])
    ## "Invert" your R_gen

    rmse = 0.5*tf.reduce_sum(tf.matmul(resvec_x,tf.transpose(resvec_x)))
    ## If we were really multiplying elementwise by a matrix with R_inv on the
    ## diagonal, it would be different.

    ## We also have a cost coming from the log determinant in the
    ## denominator:
    # denom = 0.5*tf.reduce_sum(tf.log(R_inv))*batch_size
    # prefactor = -0.5*(dim_x*dim_x*3)*np.log(2*np.pi)*batch_size
    return rmse

def likelihood_cost_vec(true_images,gen_images,gen_params):
    ## Let's impose a gaussian likelihood elementwise and call the
    ## log likelihood an RMSE:
    resvec_x = tf.reshape(true_images-gen_images,[batch_size,])
    ## "Invert" your R_gen
    R_inv = (1./gen_params['R_gen']).astype('float32')
    R_inv_constant = R_inv[0]
    rmse = -0.5*tf.reduce_sum(tf.square(resvec_x))*R_inv_constant
    ## If we were really multiplying elementwise by a matrix with R_inv on the
    ## diagonal, it would be different.

    ## We also have a cost coming from the log determinant in the
    ## denominator:
    denom = 0.5*tf.reduce_sum(tf.log(R_inv))*batch_size
    prefactor = -0.5*(imsize)*np.log(2*np.pi)*batch_size
    return denom+rmse+prefactor

def prior_cost(samples,Q_inv_gen,Q0_inv_gen,gen_params):
    ## The prior cost is defined on the latent variables. We require
    ## That they are smooth in time:

    dynres = samples[1:,:]-tf.matmul(samples[:-1,:],tf.constant(gen_params['A_gen'].T))
    rmse_dyn = -0.5*tf.reduce_sum(tf.matmul(tf.transpose(dynres),dynres)*Q_inv_gen)
    denom_dyn = 0.5*tf.reduce_sum(tf.log(tf.matrix_determinant(Q_inv_gen)))*(tf.cast(tf.shape(samples)[0],tf.float32)-1)

    ## We also have conditions on the initial configuration:
    rmse_init = -0.5*tf.matmul(samples[0:1,:],tf.matmul(Q0_inv_gen,tf.transpose(samples[0:1,:])))
    denom_init = 0.5*tf.reduce_sum(tf.log(tf.matrix_determinant(Q0_inv_gen)))
    prefactor = -0.5*(dim_z)*np.log(2*np.pi)
    total_prior = rmse_dyn+denom_dyn+rmse_init+denom_init+prefactor
    return rmse_dyn+denom_dyn+rmse_init+denom_init+prefactor

## If we add in a static term representing the static variable,
## This can be represented via a KL term that is analytic:
def KL_stat(means,vars):
    ## this is the negative "backwards" KL of p with respect to q.
    return 0.5*tf.reduce_sum(1+vars-tf.square(means)-tf.exp(vars))

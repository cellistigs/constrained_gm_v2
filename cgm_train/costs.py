## A place to store all the costs that we implement:
import tensorflow as tf
import numpy as np

# The total cost that we will use to train our model is a
# sum of two parts (as is always true of the elbo)
# We can best parse our cost by considering it as an entropic term
# (that can be evaluated analytically) and an expectation over the
# log-joint distribution of x and z that we sample via SGVB.

# The entropy in Z can be formulated as a function of the
# Cholesky decomposition of Sigma (this is R).
def entropy_cost(R,dim_z,batch_size):
    ## Entropy for a gaussian is a simple function of the covariance.
    logdet = -2*tf.reduce_sum(tf.log(tf.diag_part(R)+1e-10))
    return logdet/2+dim_z*batch_size/2*np.log(2*np.pi*np.exp(1))

# The joint distribution can further be formulated as sum of a
# log likelihood and a prior over the structure of Z (which also encodes
# dynamical information)
def likelihood_cost(true_images,gen_images,gen_params,batch_size,dim_z,dim_x):
    ## Let's impose a gaussian likelihood elementwise and call the
    ## log likelihood an RMSE:
    resvec_x = tf.reshape(true_images,[batch_size,-1])-tf.reshape(gen_images,[batch_size,-1])
    ## "Invert" your R_gen
    R_inv = (1./gen_params['R_gen']).astype('float32')
    R_inv_constant = R_inv[0]
    rmse = -0.5*tf.reduce_sum(tf.matmul(resvec_x,tf.transpose(resvec_x)))*R_inv_constant
    ## If we were really multiplying elementwise by a matrix with R_inv on the
    ## diagonal, it would be different.

    ## We also have a cost coming from the log determinant in the
    ## denominator:
    denom = 0.5*tf.reduce_sum(tf.log(R_inv))*batch_size
    prefactor = -0.5*(dim_x*dim_x*3)*np.log(2*np.pi)*batch_size
    return denom+rmse+prefactor

def regression_cost(true_images,gen_images,batch_size):
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

def likelihood_cost_vec(true_images,gen_images,gen_params,batch_size,dim_z,dim_x):
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
    prefactor = -0.5*(dim_x)*np.log(2*np.pi)*batch_size
    return denom+rmse+prefactor

def prior_cost(samples,Q_inv_gen,Q0_inv_gen,gen_params,dim_z,batch_size,dim_x):
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

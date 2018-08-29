## File to store input preprocessing scripts:
from skimage.transform import resize
from config import imsize,batch_size
import tensorflow as tf
import numpy as np
##### Move these helper functions into an input processing folder.
def halfsize(image):
    return resize(image,(imsize,imsize))
### Make input pipeline:
# Define a function that wraps the preprocessing steps:
def preprocess(serialized_example):
    featureset = {'video': tf.FixedLenFeature([],tf.string),
                  'frame': tf.FixedLenFeature([],tf.int64),
                  'image': tf.FixedLenFeature([],tf.string),
                  'intensity': tf.FixedLenFeature([],tf.float32),
                  }
    features = tf.parse_single_example(serialized_example,features = featureset)

    # Convert string representation to integers:
    image = tf.decode_raw(features['image'],tf.float64)

    image = tf.reshape(image,[128,128,3])

    image = tf.image.resize_images(image,[imsize,imsize])

    image = tf.cast(image,tf.float32)
    # image.set_shape
    # Convert label to string:
    label = features['frame']
    video = features['video']
    intensity = features['intensity']
    return image,label,video,intensity

def preprocess_p(serialized_example):
    featureset = {'video': tf.FixedLenFeature([],tf.string),
                  'frame': tf.FixedLenFeature([],tf.int64),
                  'image': tf.FixedLenFeature([],tf.string),
                  'intensity': tf.FixedLenFeature([],tf.float32),
                  'background': tf.FixedLenFeature([],tf.string)
                  }
    features = tf.parse_single_example(serialized_example,features = featureset)

    # Convert string representation to integers:
    image = tf.decode_raw(features['image'],tf.float64)

    image = tf.reshape(image,[128,128,3])

    image = tf.image.resize_images(image,[imsize,imsize])

    image = tf.cast(image,tf.float32)
    # image.set_shape
    # Convert label to string:
    label = features['frame']
    video = features['video']
    intensity = features['intensity']
    params = tf.decode_raw(features['background'],tf.float64)
    params = tf.reshape(tf.cast(params,tf.float32),[1,2])
    return image,label,video,intensity,params

def preprocess_p_m(serialized_example):
    featureset = {'video': tf.FixedLenFeature([],tf.string),
                  'frame': tf.FixedLenFeature([],tf.int64),
                  'image': tf.FixedLenFeature([],tf.string),
                  'intensity': tf.FixedLenFeature([],tf.float32),
                  'background': tf.FixedLenFeature([],tf.string),
                  'position': tf.FixedLenFeature([],tf.string),
                  }
    features = tf.parse_single_example(serialized_example,features = featureset)

    # Convert string representation to integers:
    image = tf.decode_raw(features['image'],tf.float64)

    image = tf.reshape(image,[128,128,3])

    image = tf.image.resize_images(image,[imsize,imsize])

    ## Append xy coordinate meshgrids:
    index = np.linspace(0,1,imsize)

    xx,yy = np.meshgrid(index,index)

    posgrid = tf.cast(tf.constant(np.concatenate((xx.reshape(imsize,imsize,1),yy.reshape(imsize,imsize,1)),axis = 2)),tf.float32)
    image = tf.concat((tf.cast(image,tf.float32),posgrid),axis = 2)


    # image.set_shape
    # Convert label to string:
    label = features['frame']
    video = features['video']
    intensity = features['intensity']
    params = tf.decode_raw(features['background'],tf.float64)
    params = tf.reshape(tf.cast(params,tf.float32),[1,2])
    pos = tf.decode_raw(features['position'],tf.float64)
    pos = (tf.reshape(tf.cast(pos[:2],tf.float32),[1,2])+20)/40
    return image,label,video,intensity,params,pos

def preprocess_eff(serialized_example,base_image):
    featureset = {'video': tf.FixedLenFeature([],tf.string),
                  'frame': tf.FixedLenFeature([],tf.int64),
                  'image': tf.FixedLenFeature([],tf.string),
                  'background': tf.FixedLenFeature([],tf.string),
                  }
    features = tf.parse_single_example(serialized_example,features = featureset)

    # Convert string representation to integers:
    image = tf.decode_raw(features['image'],tf.float64)

    image = tf.reshape(image,[128,128,3])

    image = tf.image.resize_images(image,[imsize,imsize])

    ## Append xy coordinate meshgrids:
    index = np.linspace(0,1,imsize)

    xx,yy = np.meshgrid(index,index)

    posgrid = tf.cast(tf.constant(np.concatenate((xx.reshape(imsize,imsize,1),yy.reshape(imsize,imsize,1)),axis = 2)),tf.float32)
    image = tf.concat((tf.cast(image,tf.float32),posgrid),axis = 2)


    # image.set_shape
    # Convert label to string:
    label = features['frame']
    video = features['video']
    intensity = features['intensity']
    params = tf.decode_raw(features['background'],tf.float64)
    params = tf.reshape(tf.cast(params,tf.float32),[1,2])
    pos = tf.decode_raw(features['position'],tf.float64)
    pos = (tf.reshape(tf.cast(pos[:2],tf.float32),[1,2])+20)/40
    return image,label,video,params

def preprocess_eff_many(serialized_example):

    featureset = {'video': tf.FixedLenFeature([],tf.string),
                  'frame': tf.FixedLenFeature([],tf.int64),
                  'image': tf.FixedLenFeature([],tf.string),
                  'background': tf.FixedLenFeature([],tf.string),
                  }
    features = tf.parse_single_example(serialized_example,features = featureset)

    # Convert string representation to integers:
    image = tf.decode_raw(features['image'],tf.float64)

    image = tf.reshape(image,[128,128,3])

    image = image[8:-8,8:-8,:]

    image = tf.image.resize_images(image,[imsize,imsize])

#     ## Append xy coordinate meshgrids:
#     index = np.linspace(0,1,imsize)

#     xx,yy = np.meshgrid(index,index)

#     posgrid = tf.cast(tf.constant(np.concatenate((xx.reshape(imsize,imsize,1),yy.reshape(imsize,imsize,1)),axis = 2)),tf.float32)
#     image = tf.concat((tf.cast(image,tf.float32),posgrid),axis = 2)


    # image.set_shape
    # Convert label to string:
    label = features['frame']
    video = features['video']
    params = tf.decode_raw(features['background'],tf.float64)
    params = tf.reshape(tf.cast(params,tf.float32),[2,])
    intensity = tf.random_uniform([1,])
#     pos = tf.decode_raw(features['position'],tf.float64)
#     pos = (tf.reshape(tf.cast(pos[:2],tf.float32),[1,2])+20)/40
    return image,label,video,params,intensity

## Now specify separate pipelines that will give us the inputs (defined with placeholders)
## to feed to an encoder network, a decoder network, or full training.
def temppipeline_0(filenames,batch_size,imsize):
    # Apply preprocessing
    base_dataset = tf.data.TFRecordDataset(filenames)
    # Get out the images and tags in a useful format
    preprocessed = base_dataset.map(preprocess).shuffle(50)
    ## We now want to batch the dataset into groups of ten neighboring frames:
    batches = preprocessed.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    ## Each of these groups of 10 will be processed together, and their latents will
    ## reflect this structure.
    ## Shuffle data:
    shuffled = batches.repeat(2).shuffle(40)
    ## Make an iterator:
    iterator = shuffled.make_initializable_iterator()
    ## We will initialize and feed through this iterator at training time.
    next_images_it,_,_,intensity_it = iterator.get_next()

    ## now feed these through iterators in case we want to name/retrieve them
    ## later:
    next_images = tf.placeholder_with_default(next_images_it,shape = [None,imsize,imsize,5],name = 'input_samples')

    return next_images_it,intensity_it


def temppipeline_1(filenames,batch_size,imsize):
    # filenames = ['datadirectory/toydynamics_nograv/Video_ball_color_small_b'+str(i)+'encoder_train.tfrecords' for i in range(10)]

    # # First try batching by background. In order to do this, we first
    # # get the tf record filenames into a dataset that can be manipulated
    # dataset_names = tf.data.Dataset.from_tensor_slices(filenames).shuffle(20)
    # ## Now we have to define ALL preprocessing that we do prior to
    # ## interleaving the datasets. This means the preprocessing defined above, as
    # ## well as shuffling (but before batching).
    # mixed = dataset_names.apply(tf.contrib.data.parallel_interleave(lambda x:tf.data.TFRecordDataset(x).map(preprocess_p).shuffle(100),cycle_length = 10,block_length=5,sloppy = True, prefetch_input_elements=50))
    # ## Having cycle and block length equivalent to shuffle length: Think about this.
    # ## Now we have to batch the mixed dataset and shuffle it:
    # final = mixed.shuffle(100).apply(tf.contrib.data.batch_and_drop_remainder(batch_size)).repeat(2).shuffle(40)
    # iterator = final.make_initializable_iterator()
    # next_images_it,_,_,intensity_it,params_it = iterator.get_next()
    base_dataset = tf.data.TFRecordDataset(filenames)
    nb_shards = 6
    index_dataset = tf.data.Dataset.range(nb_shards)
    mixed = index_dataset.apply(tf.contrib.data.parallel_interleave(lambda x:base_dataset.shard(nb_shards,x).map(preprocess_p_m).repeat(1).shuffle(1000),cycle_length = 6,block_length = 10,sloppy = True, prefetch_input_elements=2000))
    # final = mixed.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    final = mixed.batch(batch_size)
    # Get out the images and tags in a useful format
    # preprocessed = base_dataset.map(preprocess_p_m).shuffle(50)
    # ## We now want to batch the dataset into groups of ten neighboring frames:
    # batches = preprocessed.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    # ## Each of these groups of 10 will be processed together, and their latents will
    # ## reflect this structure.
    # ## Shuffle data:
    # shuffled = batches.repeat(2).shuffle(40)
    # prefetched = shuffled.prefetch(buffer_size=5)
    ## Make an iterator:
    iterator = tf.data.Iterator.from_structure(final.output_types, final.output_shapes)
    # iterator = final.make_initializable_iterator()
    next_images_it,label_it,_,intensity_it,params_it,pos_it = iterator.get_next()

    initializer = iterator.make_initializer(final,'init_op')


    return next_images_it,label_it,intensity_it,params_it,pos_it

## Pipeline for efficient image generation:
def temppipeline_2(filenames,base_image):
    # filenames = ['datadirectory/toydynamics_nograv/Video_ball_color_small_b'+str(i)+'encoder_train.tfrecords' for i in range(10)]

    # # First try batching by background. In order to do this, we first
    # # get the tf record filenames into a dataset that can be manipulated
    # dataset_names = tf.data.Dataset.from_tensor_slices(filenames).shuffle(20)
    # ## Now we have to define ALL preprocessing that we do prior to
    # ## interleaving the datasets. This means the preprocessing defined above, as
    # ## well as shuffling (but before batching).
    # mixed = dataset_names.apply(tf.contrib.data.parallel_interleave(lambda x:tf.data.TFRecordDataset(x).map(preprocess_p).shuffle(100),cycle_length = 10,block_length=5,sloppy = True, prefetch_input_elements=50))
    # ## Having cycle and block length equivalent to shuffle length: Think about this.
    # ## Now we have to batch the mixed dataset and shuffle it:
    # final = mixed.shuffle(100).apply(tf.contrib.data.batch_and_drop_remainder(batch_size)).repeat(2).shuffle(40)
    # iterator = final.make_initializable_iterator()
    # next_images_it,_,_,intensity_it,params_it = iterator.get_next()
    base_dataset = tf.data.TFRecordDataset(filenames)
    nb_shards = 6
    index_dataset = tf.data.Dataset.range(nb_shards)
    mixed = index_dataset.apply(tf.contrib.data.parallel_interleave(lambda x:base_dataset.shard(nb_shards,x).map(preprocess_eff_many).repeat(1).shuffle(1000),cycle_length = 6,block_length = 10,sloppy = True, prefetch_input_elements=1000))
    # final = mixed.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    final = mixed.batch(batch_size)
    # Get out the images and tags in a useful format
    # preprocessed = base_dataset.map(preprocess_p_m).shuffle(50)
    # ## We now want to batch the dataset into groups of ten neighboring frames:
    # batches = preprocessed.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    # ## Each of these groups of 10 will be processed together, and their latents will
    # ## reflect this structure.
    # ## Shuffle data:
    # shuffled = batches.repeat(2).shuffle(40)
    # prefetched = shuffled.prefetch(buffer_size=5)
    ## Make an iterator:


    iterator = tf.data.Iterator.from_structure(final.output_types, final.output_shapes)
    # iterator = final.make_initializable_iterator()
    next_images_it,label_it,_,params_it,intensity_it = iterator.get_next()
    intensity_mult = tf.reshape(intensity_it,shape=(batch_size,1,1,1))

    final_image= tf.maximum(tf.minimum(next_images_it+base_image*intensity_mult/255.,1),0)

    initializer = iterator.make_initializer(final,'init_op')
    # final_image = next_images_it+base_im*tf.random_uniformuniform/255.
    ## We finally

    return final_image,label_it,intensity_it,params_it,initializer

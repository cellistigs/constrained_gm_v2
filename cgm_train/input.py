## File to store input preprocessing scripts:
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

def preprocess_p_m_new(serialized_example):
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

    image = tf.reshape(image,[imsize,imsize,3])

    # image = tf.image.resize_images(image,[imsize,imsize])

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
    pos = tf.reshape(tf.cast(pos[:2],tf.float32),[1,2])
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

def preprocess_eff_many_bonly(serialized_example):

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
#     pos = tf.decode_raw(features['position'],tf.float64)
#     pos = (tf.reshape(tf.cast(pos[:2],tf.float32),[1,2])+20)/40
    return image,label,video,params

def preprocess_VAE_vanilla(serialized_example):
    featureset = {'video': tf.FixedLenFeature([],tf.string),
                  'frame': tf.FixedLenFeature([],tf.int64),
                  'image': tf.FixedLenFeature([],tf.string),
                  'mouse': tf.FixedLenFeature([],tf.int64),
                  'position':tf.FixedLenFeature([],tf.string)}

    features = tf.parse_single_example(serialized_example,features = featureset)
    image = tf.decode_raw(features['image'],tf.uint8)
    image = tf.reshape(image,[imsize,imsize,3])
    image = tf.image.resize_images(image,[imsize,imsize])
    image = tf.cast(image,tf.float32)
    mouse = features['frame']
    video = features['video']
    position = tf.decode_raw(features['position'],tf.float64)
    return image,position,mouse,video


################################################################################
## Now specify separate pipelines that will give us the inputs (defined with placeholders)
## to feed to an encoder network, a decoder network, or full training.

def VAE_pipeline(filenames,batch_size,imsize):
    base_dataset = tf.data.TFRecordDataset(filenames)
    nb_shards = 10
    index_dataset = tf.data.Dataset.range(nb_shards-6)
    mixed = index_dataset.apply(tf.contrib.data.parallel_interleave(lambda x:base_dataset.shard(nb_shards,x).map(preprocess_VAE_vanilla).take(5000).shuffle(500),cycle_length = 4,block_length = 10,sloppy = True, prefetch_input_elements=5000))
    final = mixed.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    #final = mixed.batch(batch_size)

    iterator = tf.data.Iterator.from_structure(final.output_types,final.output_shapes)
    im,position,mouse,video = iterator.get_next()
    initz = iterator.make_initializer(final,'init_op')

    return im,position,mouse,video,initz

def VAE_train_pipeline(filenames,batch_size,imsize):
    base_dataset = tf.data.TFRecordDataset(filenames)
    nb_shards = 4
    index_dataset = tf.data.Dataset.range(nb_shards)
    mixed = index_dataset.apply(tf.contrib.data.parallel_interleave(lambda x:base_dataset.shard(nb_shards,x).map(preprocess_VAE_vanilla).repeat(2).shuffle(1000),cycle_length = 4,block_length = 10,sloppy = True, prefetch_input_elements=1000))
    # final = mixed.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    final = mixed.batch(batch_size)

    iterator = tf.data.Iterator.from_structure(final.output_types,final.output_shapes)
    im,position,mouse,video = iterator.get_next()
    initz = iterator.make_initializer(final,'init_op')

    return im,position,mouse,video,initz

def VAE_traintest_pipeline(filenames_train,filenames_test,batch_size,imsize):
    base_dataset_train = tf.data.TFRecordDataset(filenames_train)
    nb_shards = 4
    index_dataset = tf.data.Dataset.range(nb_shards)
    mixed_train = index_dataset.apply(tf.contrib.data.parallel_interleave(lambda x:base_dataset_train.shard(nb_shards,x).map(preprocess_VAE_vanilla).repeat(2).shuffle(1000),cycle_length = 4,block_length = 10,sloppy = True, prefetch_input_elements=1000))
    # final = mixed.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    final_train = mixed_train.batch(batch_size)

    base_dataset_test = tf.data.TFRecordDataset(filenames_test)
    test = base_dataset_test.map(preprocess_VAE_vanilla).batch(batch_size)

    iterator = tf.data.Iterator.from_structure(final_train.output_types,final_train.output_shapes)
    im,position,mouse,video = iterator.get_next()
    initz = iterator.make_initializer(final_train,'init_op')
    init_test = iterator.make_initializer(test,'test_init_op')

    return im,position,mouse,video,initz,init_test

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

def temppipeline_1_1(filenames,batch_size,imsize):
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


    return next_images_it,label_it,intensity_it,params_it,pos_it,initializer

def newpipeline_1_1(filenames,batch_size,imsize):
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
    mixed = index_dataset.apply(tf.contrib.data.parallel_interleave(lambda x:base_dataset.shard(nb_shards,x).map(preprocess_p_m_new).repeat(1).shuffle(1000),cycle_length = 6,block_length = 10,sloppy = True, prefetch_input_elements=2000))
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


    return next_images_it,label_it,intensity_it,params_it,pos_it,initializer

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
    mixed = index_dataset.apply(tf.contrib.data.parallel_interleave(lambda x:base_dataset.shard(nb_shards,x).map(preprocess_eff_many).repeat(2).shuffle(1000),cycle_length = 6,block_length = 10,sloppy = True, prefetch_input_elements=1000))
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

def temppipeline_3(filenames,base_image):
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
    mixed = index_dataset.apply(tf.contrib.data.parallel_interleave(lambda x:base_dataset.shard(nb_shards,x).map(preprocess_eff_many).repeat(1),cycle_length = 5,block_length = 10,sloppy = True, prefetch_input_elements=900))
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

def backonlypipeline(filenames):
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
    mixed = index_dataset.apply(tf.contrib.data.parallel_interleave(lambda x:base_dataset.shard(nb_shards,x).map(preprocess_eff_many_bonly).repeat(1),cycle_length = 5,block_length = 10,sloppy = True, prefetch_input_elements=900))
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
    next_images_it,label_it,_,params_it = iterator.get_next()


    # final_image= tf.maximum(tf.minimum(next_images_it+base_image*intensity_mult/255.,1),0)

    initializer = iterator.make_initializer(final,'init_op')
    # final_image = next_images_it+base_im*tf.random_uniformuniform/255.
    ## We finally

    return next_images_it,label_it,params_it,initializer

## Pipeline for the motion input
def simpipeline1(filenames):
    base_dataset = tf.data.TFRecordDataset(filenames)
    preprocessed = base_dataset.map(preprocess_p_m)
    ## We now want to batch the dataset into groups of x neighboring frames:
    batches = preprocessed.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    ## Make an iterator without shuffling

    iterator = batches.make_initializable_iterator()
    ## now define the next set of images.
    # next_images_it,_,_,intensity_it,params_it = iterator.get_next()
    next_images_it,label_it,_,intensity_it,params_it,pos_it = iterator.get_next()

    sess.run(iterator.initializer)

    return next_images_it,label_it,intensity_it,params_it,pos_it

## Pipeline for simulating fro the efficiently stored many back dataset.
def simpipeline2(filenames,base_image):
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
    nb_shards = 10
    names = tf.data.Dataset.list_files([filenames[0] for i in range(nb_shards)])
    dataset = names.apply(tf.contrib.data.parallel_interleave(lambda file: tf.data.TFRecordDataset(file).map(preprocess_eff_many),cycle_length = nb_shards))

    # base_dataset = tf.data.TFRecordDataset([filenames for i in range(nb_shards)])

    # index_dataset = tf.data.Dataset.range(nb_shards)
    # repeated = index_dataset.apply(tf.contrib.data.parallel_interleave(lambda x:base_dataset.shard(nb_shards,x).map(preprocess_eff_many),cycle_length = nb_shards,block_length = 1))
    final = dataset.batch(batch_size)



    # base_dataset = tf.data.TFRecordDataset(filenames)
    # nb_shards = 6
    # index_dataset = tf.data.Dataset.range(nb_shards)
    # mixed = index_dataset.apply(tf.contrib.data.parallel_interleave(lambda x:base_dataset.shard(nb_shards,x).map(preprocess_eff_many).repeat(2).shuffle(1000),cycle_length = 6,block_length = 10,sloppy = True, prefetch_input_elements=1000))
    # # final = mixed.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    # final = mixed.batch(batch_size)

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

    intensity_mult = np.tile(np.linspace(0,1,nb_shards).reshape(nb_shards,1,1,1),(int(batch_size/nb_shards),1,1,1))
    intensity_it = tf.constant(intensity_mult.reshape(batch_size,1)) ## sorry
    final_image= tf.maximum(tf.minimum(next_images_it+base_image*intensity_mult/255.,1),0)

    initializer = iterator.make_initializer(final,'init_op')
    # final_image = next_images_it+base_im*tf.random_uniformuniform/255.
    ## We finally

    return final_image,label_it,intensity_it,params_it,initializer

## Pipeline for simulating fro the efficiently stored many back dataset.
def simpipelinemback(filenames):
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
    nb_shards = 10
    names = tf.data.Dataset.list_files([filenames[0] for i in range(nb_shards)])
    dataset = names.apply(tf.contrib.data.parallel_interleave(lambda file: tf.data.TFRecordDataset(file).map(preprocess_eff_many_bonly),cycle_length = nb_shards))

    # base_dataset = tf.data.TFRecordDataset([filenames for i in range(nb_shards)])

    # index_dataset = tf.data.Dataset.range(nb_shards)
    # repeated = index_dataset.apply(tf.contrib.data.parallel_interleave(lambda x:base_dataset.shard(nb_shards,x).map(preprocess_eff_many),cycle_length = nb_shards,block_length = 1))
    final = dataset.batch(batch_size)



    # base_dataset = tf.data.TFRecordDataset(filenames)
    # nb_shards = 6
    # index_dataset = tf.data.Dataset.range(nb_shards)
    # mixed = index_dataset.apply(tf.contrib.data.parallel_interleave(lambda x:base_dataset.shard(nb_shards,x).map(preprocess_eff_many).repeat(2).shuffle(1000),cycle_length = 6,block_length = 10,sloppy = True, prefetch_input_elements=1000))
    # # final = mixed.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    # final = mixed.batch(batch_size)

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
    next_images_it,label_it,_,params_it = iterator.get_next()

    initializer = iterator.make_initializer(final,'init_op')
    # final_image = next_images_it+base_im*tf.random_uniformuniform/255.
    ## We finally

    return next_images_it,label_it,params_it,initializer

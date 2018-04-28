#!/usr/bin/env python3
# coding: utf-8

# In[1]:

import warnings
warnings.filterwarnings('ignore')
import os
import scipy
import scipy.io as sio
#from scipy import ndimage
from skimage import io,data,color
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# In[2]:

flags = tf.app.flags
flags.DEFINE_string('path', 'test_images/image1.jpg', 'the path for image')
cfg = tf.app.flags.FLAGS
#image = np.array(ndimage.imread(cfg.path, flatten=False)).reshape((1, 48, 48, 1)).astype(np.float)
#teX = tf.convert_to_tensor(image.astype(np.float) / 255., tf.float32)

img_name=cfg.path
img=io.imread(img_name,as_grey=False)
img_grey = color.rgb2gray(img)
image = scipy.misc.imresize(img_grey, size=(48,48)).reshape(1,48,48,1)
teX = tf.convert_to_tensor(image.astype(np.float) / 255., tf.float32)

def load_TFD(path, fold, is_training, data_augmentation = False):

    #load 4178 images, flods, labels
    data = sio.loadmat(os.path.join(path, 'TFD_48.mat'))
    images = data['images'].reshape((4178, 48, 48, 1)).astype(np.float)
    folds = data['folds']
    labels = data['labs_ex'].reshape((4178)).astype(np.int32) - int(1)
    
    #one hot
    #labels = tf.one_hot(labels, depth=7, axis=1, dtype=tf.float32)
    if (is_training and data_augmentation):
        
        training = (folds[:,fold] == 1)
        training_images = images[training,:,:]
        training_labels = labels[training]
        num = training_images.shape[0]
        origImages = []
        fliped_images = []
        fliped_images_labels = np.empty((num,), dtype="int32")
        brighted_contrasted_images = []
        brighted_contrasted_images_labels = np.empty((num,), dtype="int32")

        for i in range(num):
            origImage = tf.cast(training_images[i], tf.float32)
            origImages.append(origImage)
            fliped_image = tf.image.flip_left_right(origImage)
            fliped_images.append(fliped_image)
            fliped_images_labels[i] = training_labels[i]
            brighted_image = tf.image.random_brightness(origImage, max_delta=0.4)
            brighted_contrasted_image = tf.image.random_contrast(brighted_image, lower=0.2, upper=1.8)
            brighted_contrasted_images.append(brighted_contrasted_image)
            brighted_contrasted_images_labels[i] = training_labels[i]
            
        trX = tf.concat([origImages, fliped_images, brighted_contrasted_images],0)
        trX = trX / 255.
        
        trY = np.append([training_labels, fliped_images_labels], brighted_contrasted_images_labels)
       
        print(trX.shape)
        print(trY.shape)
        return trX,trY
    
    elif is_training:
        
        training = (folds[:,fold] == 1)
        trX = tf.convert_to_tensor(images[training,:,:] / 255., tf.float32)
        trY = labels[training]
        print(trX.shape)
        print(trY.shape)
        return trX,trY
    
    else:
        
        test = (folds[:,fold] == 3)
        teX = tf.convert_to_tensor(images[test,:,:].astype(np.float) / 255., tf.float32)
        teY = tf.convert_to_tensor(labels[test], tf.int32)
        #print(teX.shape)
        #print(teY.shape)
        return teX,teY

def get_batch_data(batch_size):
    trX, trY = load_TFD('data', 0, True, True)

    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=8,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)
    return(X, Y)


# In[3]:


#batch norm
def conv_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)
    
    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)
    
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name="moments")
    #def moments(x, axes, name=None, keep_dims=False)
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = control_flow_ops.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var))
    
    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, True)
    #def batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None)
    return normed

def layer_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)

    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = control_flow_ops.cond(phase_train,
        mean_var_with_update,
        lambda: (ema_mean, ema_var))

    reshaped_x = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var,
        beta, gamma, 1e-3, True)
    return tf.reshape(normed, [-1, n_out])

def filter_summary(V, weight_shape):
    ix = weight_shape[0]
    iy = weight_shape[1]
    cx, cy = 8, 8
    V_T = tf.transpose(V, (3, 0, 1, 2))
    tf.summary.image("filters", V_T, max_outputs=64)
    
def conv2d(input, weight_shape, bias_shape, phase_train, s=1, visualizer=False):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0 / incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    if visualizer:
        filter_summary(W, weight_shape)
    bias_init = tf.zeros_initializer()
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    
    logits = tf.nn.conv2d(input, filter=W, strides=[1,s,s,1], padding='SAME') + b
    return tf.nn.relu(conv_batch_norm(logits, weight_shape[3], phase_train))

#only have linear part, don't have activation part
def layer(input, weight_shape, bias_shape, phase_train, visualizer=False):
    weight_init = tf.random_normal_initializer(stddev=(2.0 / weight_shape[0])**0.5)
    bias_init = tf.zeros_initializer()
    W = tf.get_variable('W', weight_shape, initializer=weight_init)
    if visualizer:
        filter_summary(W, weight_shape)
    b = tf.get_variable('b', bias_shape, initializer=bias_init)
    
    logits = tf.matmul(input, W) + b
    return layer_batch_norm(logits, weight_shape[1], phase_train)

def inference(x, phase_train):
    with tf.variable_scope("convolution_1"):
        conv_1 = conv2d(x, [3,3,1,96], [96], phase_train, visualizer=False)
    with tf.variable_scope("convolution_2"):
        conv_2 = conv2d(conv_1, [3,3,96,96], [96], phase_train, visualizer=False)
    with tf.variable_scope("pseudopool_1"):
        conv_3 = conv2d(conv_2, [3,3,96,96], [96], phase_train, s=2)
    with tf.variable_scope("convolution_4"):
        conv_4 = conv2d(conv_3, [3,3,96,192],[192], phase_train, visualizer=False)
    with tf.variable_scope("convolution_5"):
        conv_5 = conv2d(conv_4, [3,3,192,192],[192], phase_train, visualizer=False)
    with tf.variable_scope("pseudopool_2"):
        conv_6 = conv2d(conv_5, [3,3,192,192],[192], phase_train, s=2, visualizer=False)
    with tf.variable_scope("convolution_7"):
        conv_7 = conv2d(conv_6, [3,3,192,192],[192],phase_train)
    with tf.variable_scope("convolution_8"):
        conv_8 = conv2d(conv_7, [1,1,192,192],[192],phase_train)
    with tf.variable_scope("convolution_9"):
        conv_9 = conv2d(conv_8, [1,1,192,7],[7],phase_train)
    with tf.variable_scope("global_average"):
        average = tf.nn.avg_pool(conv_9, ksize=[1,conv_9.shape[1],conv_9.shape[2],1], strides=[1,1,1,1],padding="VALID")
    with tf.variable_scope("softmax_linear"):
        dim = 1
        for d in average.get_shape()[1:].as_list():
            dim *= d
        logits = tf.reshape(average, [-1, dim])
    return logits


def evaluate(logits, y):
    
    preds = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    tf.summary.scalar("validation_error", (1.0 - accuracy))
    return accuracy

def show_emotion(result):
    if result == 0:
        print('Prediction: Annoyed')
    elif result == 1:
        print('Prediction: Disgust')
    elif result == 2:
        print('Prediction: Fear')
    elif result == 3:
        print('Prediction: Happy')
    elif result == 4:
        print('Prediction: Sad')
    elif result == 5:
        print('Prediction: Surprise')
    else:
        print('Prediction: Neutral')



with tf.variable_scope("TFD_conv_model"):
        
    with tf.name_scope('data'):
        X = tf.placeholder(tf.float32, [None, 48, 48, 1], name="X_placeholder")
        #Y = tf.placeholder(tf.int32, [None], name="Y_placeholder")
        phase_train = tf.placeholder(tf.bool, name="phase_train")
        
    #val_images, val_labels = load_TFD('data', 0, False)
    
    logits = inference(X, phase_train)
        
    #global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    #eval_op = evaluate(logits, Y)

    preds = tf.nn.softmax(logits)
    prediction = tf.cast(tf.argmax(preds, 1), tf.int32)

            
    with tf.name_scope('summary'):
        summary_op = tf.summary.merge_all()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess)
            
        saver = tf.train.Saver()
        #ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/emotion/checkpoint'))
        ckpt_path = 'checkpoints/emotion/TFD-convnet-7098'
        #saver.restore(sess, ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt_path)

        start_time = time.time()

        #val_x, val_y = sess.run([val_images, val_labels])
        val_x = sess.run(teX)

        result = sess.run(prediction, feed_dict={X: val_x, phase_train: False})
        #accuracy = sess.run(eval_op, feed_dict={X: val_x, Y: val_y, phase_train: False})
        show_emotion(result)
        print("Total time: {0:.3} seconds".format(time.time() - start_time))

# In[ ]:




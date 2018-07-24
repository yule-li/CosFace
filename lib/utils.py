"""Functions for building the face recognition network.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
from subprocess import Popen, PIPE
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import time
import tensorflow.contrib.slim as slim
import pickle
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from tensorflow.python.platform import gfile

def py_func(func, inp, Tout, stateful = True, name=None, grad_func=None):
    rand_name = 'PyFuncGrad' + str(np.random.randint(0,1E+8))
    tf.RegisterGradient(rand_name)(grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({'PyFunc':rand_name}):
        return tf.py_func(func,inp,Tout,stateful=stateful, name=name)




def coco_forward(xw, y, m, name=None):
    #pdb.set_trace()
    xw_copy = xw.copy()
    num = len(y)
    orig_ind = range(num)
    xw_copy[orig_ind,y] -= m
    return xw_copy

def coco_help(grad,y):
    grad_copy = grad.copy()
    return grad_copy

def coco_backward(op, grad):
    
    y = op.inputs[1]
    m = op.inputs[2]
    grad_copy = tf.py_func(coco_help,[grad,y],tf.float32)
    return grad_copy,y,m

def coco_func(xw,y,m, name=None):
    with tf.op_scope([xw,y,m],name,"Coco_func") as name:
        coco_out = py_func(coco_forward,[xw,y,m],tf.float32,name=name,grad_func=coco_backward)
        return coco_out

def cos_loss(x, y,  num_cls, reuse=False, alpha=0.25, scale=64,name = 'cos_loss'):
    '''
    x: B x D - features
    y: B x 1 - labels
    num_cls: 1 - total class number
    alpah: 1 - margin
    scale: 1 - scaling paramter
    '''
    # define the classifier weights
    xs = x.get_shape()
    with tf.variable_scope('centers_var',reuse=reuse) as center_scope:
        w = tf.get_variable("centers", [xs[1], num_cls], dtype=tf.float32, 
            initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
   
    #normalize the feature and weight
    #(N,D)
    x_feat_norm = tf.nn.l2_normalize(x,1,1e-10)
    #(D,C)
    w_feat_norm = tf.nn.l2_normalize(w,0,1e-10)
    
    # get the scores after normalization 
    #(N,C)
    xw_norm = tf.matmul(x_feat_norm, w_feat_norm)  
    #implemented by py_func
    #value = tf.identity(xw)
    #substract the marigin and scale it
    value = coco_func(xw_norm,y,alpha) * scale

    #implemented by tf api
    #margin_xw_norm = xw_norm - alpha
    #label_onehot = tf.one_hot(y,num_cls)
    #value = scale*tf.where(tf.equal(label_onehot,1), margin_xw_norm, xw_norm)

    
    # compute the loss as softmax loss
    cos_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=value))

    return cos_loss 


def softmax_loss(prelogits,labels,nrof_classes,weight_decay,reuse):
    logits = slim.fully_connected(prelogits, nrof_classes, activation_fn=None, 
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
        weights_regularizer=slim.l2_regularizer(weight_decay),
        scope='softmax', reuse=reuse)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean




def contrastive_index(dists, labels, alfa):
    #pdb.set_trace()
    base_inds = np.array(range(dists.shape[0]))
    nrof_classes = dists.shape[1]
    same_indexs = base_inds*nrof_classes + labels
    dists = dists.reshape([-1])
    #dist_same = dists[same_indexs].copy()
    dists_copy = dists.copy()
    dists_copy[same_indexs] = np.NaN
    diff_indexs = np.where(dists_copy < alfa)[0]
    #pdb.set_trace()
    #dists[same_indexs] = 
    print('same indexs: {}, diff indexs: {}'.format(len(same_indexs),len(diff_indexs)))
    return np.array(same_indexs,dtype=np.int64), np.array(diff_indexs, dtype=np.int64)


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def shuffle_examples(image_paths, labels):
    shuffle_list = list(zip(image_paths, labels))
    random.shuffle(shuffle_list)
    image_paths_shuff, labels_shuff = zip(*shuffle_list)
    return image_paths_shuff, labels_shuff

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_image(file_contents, channels=3)
    return example, label
  
def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')
  
def read_and_augment_data(image_list, label_list, image_size, batch_size, max_nrof_epochs, 
        random_crop, random_flip, random_rotate, nrof_preprocess_threads, shuffle=True):
    
    images = ops.convert_to_tensor(image_list, dtype=tf.string)
    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
    
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
        num_epochs=max_nrof_epochs, shuffle=shuffle)

    images_and_labels = []
    for _ in range(nrof_preprocess_threads):
        image, label = read_images_from_disk(input_queue)
        if random_rotate:
            image = tf.py_func(random_rotate_image, [image], tf.uint8)
        if random_crop:
            image = tf.random_crop(image, [image_size, image_size, 3])
        else:
            image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        if random_flip:
            image = tf.image.random_flip_left_right(image)
        #pylint: disable=no-member
        image.set_shape((image_size, image_size, 3))
        image = tf.image.per_image_standardization(image)
        images_and_labels.append([image, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels, batch_size=batch_size,
        capacity=4 * nrof_preprocess_threads * batch_size,
        allow_smaller_final_batch=True)
  
    return image_batch, label_batch
  
def _add_loss_summaries(total_loss):
    """Add summaries for losses.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
  
    return loss_averages_op

def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
    
        grads = opt.compute_gradients(total_loss, update_gradient_vars)
        
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
   
    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
  
    return train_op
def compute_gradient(total_loss, global_step, optimizer, learning_rate, update_gradient_vars):

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
    
        grads = opt.compute_gradients(total_loss, update_gradient_vars)
        
    return opt,grads
def get_opt(optimizer,learning_rate):

    # Compute gradients.
    if optimizer=='ADAGRAD':
        opt = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer=='ADADELTA':
        opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    elif optimizer=='ADAM':
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
    elif optimizer=='RMSPROP':
        opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    elif optimizer=='MOM':
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    elif optimizer=='SGD':
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Invalid optimization algorithm')

        
    return opt



def average_gradients(tower_grads):
    '''
    Calculate the average gradient for each shared variable across all towers
    '''
    average_grads = []
    #pdb.set_trace()
    for grad_and_vars in zip(*tower_grads):
        #tower_grads[i] is the gradient computed in tower i, which have form ((g_var0,var0),...)
        #so zip(*tower_grads) has form (((grad0_gpu0,var0_gpu0),...),...)
        #this is to say: grad_and_vars is about the gradient and vars in variable i in all towers, which is gathered in a list.

        #pdb.set_trace()
        grads = []
        for g,_ in grad_and_vars:
            #print(g)
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)
        grad = tf.concat(grads,0)
        grad = tf.reduce_mean(grad,0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad,v)
        average_grads.append(grad_and_var)
    return average_grads
def sum_gradients(tower_grads):
    '''
    Calculate the average gradient for each shared variable across all towers
    '''
    average_grads = []
    #pdb.set_trace()
    for grad_and_vars in zip(*tower_grads):
        #tower_grads[i] is the gradient computed in tower i, which have form ((g_var0,var0),...)
        #so zip(*tower_grads) has form (((grad0_gpu0,var0_gpu0),...),...)
        #this is to say: grad_and_vars is about the gradient and vars in variable i in all towers, which is gathered in a list.

        #pdb.set_trace()
        grads = []
        for g,v in grad_and_vars:
            #print(g)
            if 'Batch' in v.name and 'mean' in v.name:
                print(grad_and_vars)
            
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)
        grad = tf.concat(grads,0)
        grad = tf.reduce_sum(grad,0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad,v)
        average_grads.append(grad_and_var)
    return average_grads


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image
  
def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
  

def load_data(image_paths, do_random_crop, do_flip, image_height,image_width, do_prewhiten=True,src_size=None):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_height, image_width, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if src_size is not None:
            img = misc.imresize(img,(src_size[0],src_size[1]))
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        else:
            img = img - 127.5
            img = img / 128.

        #img = crop(img, do_random_crop, image_size)
        #img = flip(img, do_random_flip)
        if do_flip:
            img = np.fliplr(img)
        images[i,:,:,:] = img
    return images

def l2_normalize(x):
    n,e = x.shape
    mean = np.mean(x,axis=1)
    mean = mean.reshape((n,1))
    mean = np.repeat(mean,e,axis=1)
    x -= mean
    norm = np.linalg.norm(x,axis=1)
    norm = norm.reshape((n,1))
    norm = np.repeat(norm,e,axis=1)
    y = np.multiply(x,1/norm)
    return y





def data_from_mx(inputs,image_size,do_resize=True,do_prewhiten=True):
    '''
    convet data in mxnet format to tf format
    params:
    - inputs: [n,h,w,c]
    '''
    nrof_samples = len(inputs)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = inputs[i].transpose((1,2,0))
        #plt.imshow(img)
        #plt.show()
        #pdb.set_trace()
        if img.ndim == 2:
            img = to_rgb(img)
        if do_resize:
            img = misc.imresize(img,[image_size,image_size])
        if do_prewhiten:
            img = prewhiten(img)
        #img = crop(img, do_random_crop, image_size)
        #img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images

def load_data_simple(image_paths,image_size,do_random_crop=False):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        #pdb.set_trace()
        if img.ndim == 2:
            img = to_rgb(img)
        #if do_prewhiten:
        #img = misc.imresize(img,[image_size,image_size])
        img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        #img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images
def load_data_caffe(image_paths,image_size):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        #pdb.set_trace()
        if img.ndim == 2:
            img = to_rgb(img)
        #if do_prewhiten:
        img = misc.imresize(img,[image_size,image_size])
        #img = crop(img, do_random_crop, image_size)
        #img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images




def get_label_batch(label_data, batch_size, batch_index):
    nrof_examples = np.size(label_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = label_data[j:j+batch_size]
    else:
        x1 = label_data[j:nrof_examples]
        x2 = label_data[0:nrof_examples-j]
        batch = np.vstack([x1,x2])
    batch_int = batch.astype(np.int64)
    return batch_int

def get_batch(image_data, batch_size, batch_index):
    nrof_examples = np.size(image_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = image_data[j:j+batch_size,:,:,:]
    else:
        x1 = image_data[j:nrof_examples,:,:,:]
        x2 = image_data[0:nrof_examples-j,:,:,:]
        batch = np.vstack([x1,x2])
    batch_float = batch.astype(np.float32)
    return batch_float

def get_triplet_batch(triplets, batch_index, batch_size):
    ax, px, nx = triplets
    a = get_batch(ax, int(batch_size/3), batch_index)
    p = get_batch(px, int(batch_size/3), batch_index)
    n = get_batch(nx, int(batch_size/3), batch_index)
    batch = np.vstack([a, p, n])
    return batch

def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
  
def get_dataset(paths, has_class_directories=True):
    dataset = []
    count = 1
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            image_paths = get_image_paths(facedir)
            class_name = '{}_{:08d}'.format(class_name,count)
            count += 1
            dataset.append(ImageClass(class_name, image_paths))
  
    return dataset
def dataset_from_list(data_dir,list_file):
    dataset = []
    lines = open(list_file,'r').read().strip().split('\n')
    path_exp = os.path.expanduser(data_dir)
    count = 1
    class_paths = {}
    for line in lines:
        image_path, _ = line.split(' ')
        class_name, _ = image_path.split('/')
        if class_name not in class_paths:
            class_paths[class_name] = []
        full_image_path = os.path.join(path_exp,image_path)
        assert os.path.exists(full_image_path), 'file {} not exist'.format(full_image_path)
        class_paths[class_name].append(full_image_path)
    dataset = []
    keys = class_paths.keys()
    keys.sort()
    for key in keys:
        dataset.append(ImageClass(key,class_paths[key]))
    return dataset

        


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths
  
def split_dataset(dataset, split_ratio, mode):
    if mode=='SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*split_ratio))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode=='SPLIT_IMAGES':
        train_set = []
        test_set = []
        min_nrof_images = 2
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            split = int(round(len(paths)*split_ratio))
            if split<min_nrof_images:
                continue  # Not enough images for test set. Skip class...
            train_set.append(ImageClass(cls.name, paths[0:split]))
            test_set.append(ImageClass(cls.name, paths[split:-1]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set

def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
    
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    indices = np.arange(nrof_pairs)
    #pdb.set_trace() 
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
    tpr = np.mean(tprs,0)
    fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


  
def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
    
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
  
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def store_revision_info(src_path, output_dir, arg_string):
  
    # Get git hash
    gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout = PIPE, cwd=src_path)
    (stdout, _) = gitproc.communicate()
    git_hash = stdout.strip()
  
    # Get local changes
    gitproc = Popen(['git', 'diff', 'HEAD'], stdout = PIPE, cwd=src_path)
    (stdout, _) = gitproc.communicate()
    git_diff = stdout.strip()
    
    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)

def list_variables(filename):
    reader = training.NewCheckpointReader(filename)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    return names

def put_images_on_grid(images, shape=(16,8)):
    nrof_images = images.shape[0]
    img_size = images.shape[1]
    bw = 3
    img = np.zeros((shape[1]*(img_size+bw)+bw, shape[0]*(img_size+bw)+bw, 3), np.float32)
    for i in range(shape[1]):
        x_start = i*(img_size+bw)+bw
        for j in range(shape[0]):
            img_index = i*shape[0]+j
            if img_index>=nrof_images:
                break
            y_start = j*(img_size+bw)+bw
            img[x_start:x_start+img_size, y_start:y_start+img_size, :] = images[img_index, :, :, :]
        if img_index>=nrof_images:
            break
    return img

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in vars(args).iteritems():
            f.write('%s: %s\n' % (key, str(value)))

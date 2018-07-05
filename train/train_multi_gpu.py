
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import os
import time
import sys
sys.path.insert(0,'lib')
sys.path.insert(0,'networks')
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.contrib import slim
import tensorflow.contrib.data as tf_data
from collections import Counter
import numpy as np
import importlib
import itertools
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1, resnet_v2
import argparse
import utils
import sphere_network as network
import inception_resnet_v1 as inception_net
import resface as resface
#import lfw
import pdb
#import cv2
#import pylab as plt

debug = False
softmax_ind = 0

from tensorflow.python.ops import data_flow_ops

def _from_tensor_slices(tensors_x,tensors_y):
    #return TensorSliceDataset((tensors_x,tensors_y))
    return tf_data.Dataset.from_tensor_slices((tensors_x,tensors_y))



def main(args):
  
    #network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    utils.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    utils.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)

    #train_set = utils.get_dataset(args.data_dir)
    train_set = utils.dataset_from_list(args.data_dir,args.list_file)
    nrof_classes = len(train_set)
    print('nrof_classes: ',nrof_classes)
    image_list, label_list = utils.get_image_paths_and_labels(train_set)
    print('total images: ',len(image_list))
    image_list = np.array(image_list)
    label_list = np.array(label_list,dtype=np.int32)

    dataset_size = len(image_list)
    single_batch_size = args.people_per_batch*args.images_per_person
    indices = range(dataset_size)
    np.random.shuffle(indices)

    def _sample_people_softmax(x):
        global softmax_ind
        if softmax_ind >= dataset_size:
            np.random.shuffle(indices)
            softmax_ind = 0
        true_num_batch = min(single_batch_size,dataset_size - softmax_ind)

        sample_paths = image_list[indices[softmax_ind:softmax_ind+true_num_batch]]
        sample_labels = label_list[indices[softmax_ind:softmax_ind+true_num_batch]]

        softmax_ind += true_num_batch

        return (np.array(sample_paths), np.array(sample_labels,dtype=np.int32))

    def _sample_people(x):
        '''We sample people based on tf.data, where we can use transform and prefetch.

        '''
    
        image_paths, num_per_class = sample_people(train_set,args.people_per_batch*(args.num_gpus-1),args.images_per_person)
        labels = []
        for i in range(len(num_per_class)):
            labels.extend([i]*num_per_class[i])
        return (np.array(image_paths),np.array(labels,dtype=np.int32))

    def _parse_function(filename,label):
        file_contents = tf.read_file(filename)
        image = tf.image.decode_image(file_contents, channels=3)
        #image = tf.image.decode_jpeg(file_contents, channels=3)
        print(image.shape)
        
        if args.random_crop:
            print('use random crop')
            image = tf.random_crop(image, [args.image_size, args.image_size, 3])
        else:
            print('Not use random crop')
            #image.set_shape((args.image_size, args.image_size, 3))
            image.set_shape((None,None, 3))
            image = tf.image.resize_images(image, size=(args.image_height, args.image_width))
            #print(image.shape)
        if args.random_flip:
            image = tf.image.random_flip_left_right(image)

        #pylint: disable=no-member
        #image.set_shape((args.image_size, args.image_size, 3))
        image.set_shape((args.image_height, args.image_width, 3))
        if debug:
            image = tf.cast(image,tf.float32)
        else:
            image = tf.cast(image,tf.float32)
            image = tf.subtract(image,127.5)
            image = tf.div(image,128.)
            #image = tf.image.per_image_standardization(image)
        return image, label

    
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))
    
        
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False,name='global_step')

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        
        
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        
                
        
        #the image is generated by sequence
        with tf.device("/cpu:0"):
                    
            softmax_dataset = tf_data.Dataset.range(args.epoch_size*args.max_nrof_epochs*100)
            softmax_dataset = softmax_dataset.map(lambda x: tf.py_func(_sample_people_softmax,[x],[tf.string,tf.int32]))
            softmax_dataset = softmax_dataset.flat_map(_from_tensor_slices)
            softmax_dataset = softmax_dataset.map(_parse_function,num_threads=8,output_buffer_size=2000)
            softmax_dataset = softmax_dataset.batch(args.num_gpus*single_batch_size)
            softmax_iterator = softmax_dataset.make_initializable_iterator()
            softmax_next_element = softmax_iterator.get_next()
            softmax_next_element[0].set_shape((args.num_gpus*single_batch_size, args.image_height,args.image_width,3))
            softmax_next_element[1].set_shape(args.num_gpus*single_batch_size)
            batch_image_split = tf.split(softmax_next_element[0],args.num_gpus)
            batch_label_split = tf.split(softmax_next_element[1],args.num_gpus)
            

    
        
        
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        print('Using optimizer: {}'.format(args.optimizer))
        if args.optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif args.optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate,0.9)
        elif args.optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        else:
            raise Exception("Not supported optimizer: {}".format(args.optimizer))
        tower_losses = []
        tower_cross = []
        tower_dist = []
        tower_reg= []
        for i in range(args.num_gpus):
            with tf.device("/gpu:" + str(i)):
                with tf.name_scope("tower_" + str(i)) as scope:
                  with slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0"):
                    with tf.variable_scope(tf.get_variable_scope()) as var_scope:
                        reuse = False if i ==0 else True
                        #with slim.arg_scope(resnet_v2.resnet_arg_scope(args.weight_decay)):
                                #prelogits, end_points = resnet_v2.resnet_v2_50(batch_image_split[i],is_training=True,
                                #        output_stride=16,num_classes=args.embedding_size,reuse=reuse)
                        #prelogits, end_points = network.inference(batch_image_split[i], args.keep_probability, 
                        #    phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, 
                        #    weight_decay=args.weight_decay, reuse=reuse)
                        if args.network ==  'sphere_network':
                            prelogits = network.infer(batch_image_split[i],args.embedding_size)
                            print(prelogits)
                        elif args.network == 'resface':
                            prelogits, _ = resface.inference(batch_image_split[i],1.0,bottleneck_layer_size=args.embedding_size,weight_decay=args.weight_decay,reuse=reuse)
                        elif args.network == 'inception_net':
                            prelogits, endpoints = inception_net.inference(batch_image_split[i],1,phase_train=True,bottleneck_layer_size=args.embedding_size,weight_decay=args.weight_decay,reuse=reuse)
                            print(prelogits)

                        elif args.network == 'resnet_v2':
                            with slim.arg_scope(resnet_v2.resnet_arg_scope(args.weight_decay)):
                                prelogits, end_points = resnet_v2.resnet_v2_50(batch_image_split[i],is_training=True,
                                        output_stride=16,num_classes=args.embedding_size,reuse=reuse)
                                prelogits = tf.squeeze(prelogits,axis=[1,2])

                        else:
                            raise Exception("Not supported network: {}".format(args.network))
                        if args.fc_bn: 

                            prelogits = slim.batch_norm(prelogits, is_training=True, decay=0.997,epsilon=1e-5,scale=True,updates_collections=tf.GraphKeys.UPDATE_OPS,reuse=reuse,scope='softmax_bn')
                        if args.loss_type == 'softmax':
                            cross_entropy_mean = utils.softmax_loss(prelogits,batch_label_split[i], len(train_set),args.weight_decay,reuse)
                            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                            tower_cross.append(cross_entropy_mean)
                            #loss = cross_entropy_mean + args.weight_decay*tf.add_n(regularization_losses)
                            loss = cross_entropy_mean + tf.add_n(regularization_losses)
                            #tower_dist.append(0)
                            #tower_cross.append(cross_entropy_mean)
                            #tower_th.append(0)
                            tower_losses.append(loss)
                            tower_reg.append(regularization_losses)
                        elif args.loss_type == 'cosface':
                            label_reshape = tf.reshape(batch_label_split[i],[single_batch_size])
                            label_reshape = tf.cast(label_reshape,tf.int64)
                            coco_loss = utils.cos_loss(prelogits,label_reshape, len(train_set),reuse,alpha=args.alpha,scale=args.scale)
                            #scatter_loss, _ = facenet.coco_loss(prelogits,label_reshape, len(train_set),reuse,alpha=args.alpha,scale=args.scale)
                            #coco_loss = scatter_loss['loss_total']
                            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                            if args.network == 'sphere_network':
                                print('reg loss using weight_decay * tf.add_n')
                                reg_loss  = args.weight_decay*tf.add_n(regularization_losses)
                            else:
                                print('reg loss using tf.add_n')
                                reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                            loss = coco_loss + reg_loss
                            
                            tower_losses.append(loss)
                            tower_reg.append(reg_loss)

                        #loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
                        tf.get_variable_scope().reuse_variables()
        total_loss = tf.reduce_mean(tower_losses)
        total_reg = tf.reduce_mean(tower_reg)
        losses = {}
        losses['total_loss'] = total_loss
        losses['total_reg'] = total_reg

        grads = opt.compute_gradients(total_loss,tf.trainable_variables(),colocate_gradients_with_ops=True)
        apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.group(apply_gradient_op)

        save_vars = [var for var in tf.global_variables() if 'Adagrad' not in var.name and 'global_step' not in var.name]
         
        #saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        saver = tf.train.Saver(save_vars, max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))        

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})

        #sess.run(iterator.initializer)
        sess.run(softmax_iterator.initializer)

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        
        with sess.as_default():
            #pdb.set_trace()

            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))

            # Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                if debug:
                    debug_train(args, sess, train_set, epoch, image_batch_gather, enqueue_op,batch_size_placeholder, image_batch_split,image_paths_split,num_per_class_split,
                            image_paths_placeholder,image_paths_split_placeholder, labels_placeholder, labels_batch, num_per_class_placeholder,num_per_class_split_placeholder,len(gpus))
                # Train for one epoch
                train(args, sess, epoch, 
                     learning_rate_placeholder, phase_train_placeholder, global_step, 
                     losses, train_op, summary_op, summary_writer, args.learning_rate_schedule_file)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

    return model_dir

def train(args, sess, epoch, 
          learning_rate_placeholder, phase_train_placeholder, global_step, 
          loss, train_op, summary_op, summary_writer, learning_rate_schedule_file):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = utils.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        start_time = time.time()
        
        print('Running forward pass on sampled images: ', end='')
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True}
        start_time = time.time()
        total_err, reg_err, _, step = sess.run([loss['total_loss'], loss['total_reg'], train_op, global_step ], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tTotal Loss %2.3f\tReg Loss %2.3f, lr %2.5f' %
                  (epoch, batch_number+1, args.epoch_size, duration, total_err, reg_err, lr))

        batch_number += 1
    return step
 

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  
  
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
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='logs/facenet_ms_mp')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='models/facenet_ms_mp')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=.9)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--loss_type', type=str,
        help='Which type loss to be used.',default='softmax')
    parser.add_argument('--network', type=str,
        help='which network is used to extract feature.',default='resnet50')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--list_file', type=str,
        help='Image list file')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--image_src_size', type=int,
        help='Src Image size (height, width) in pixels.', default=256)
    parser.add_argument('--image_height', type=int,
        help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--image_width', type=int,
        help='Image size (height, width) in pixels.', default=96)
    parser.add_argument('--people_per_batch', type=int,
        help='Number of people per batch.', default=30)
    parser.add_argument('--num_gpus', type=int,
        help='Number of gpus.', default=4)
    parser.add_argument('--images_per_person', type=int,
        help='Number of images per person.', default=5)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=600)
    parser.add_argument('--alpha', type=float,
        help='Margin for cos margin.', default=0.15)
    parser.add_argument('--scale', type=float,
        help='Scale as the fixed norm of weight and feature.', default=64.)
    parser.add_argument('--weight', type=float,
        help='weiht to balance the dist and th loss.', default=3.)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=256)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--fc_bn', 
        help='Wheater use bn after fc.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM','SGD'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')

    
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

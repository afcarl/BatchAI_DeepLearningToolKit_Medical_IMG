## Required for Azure Data Lake Store filesystem management
from azure.datalake.store import core, lib, multithread
##--Option1(SignIn): User Acct, interactively sign in to AzDataLake Store, storebotdatalakestore
#
##subscriptionId = '851da8fc-5b5f-48f2-9e14-395ce8ace4bf'
adlsAccountName = 'storebotdatalakestore'
adlCreds = lib.auth(tenant_id = '4dced229-4c95-476d-b76b-34d306d723eb', resource = 'https://datalake.azure.net/')

##--Create a filesystem client object
adlsFileSystemClient = core.AzureDLFileSystem(adlCreds, store_name=adlsAccountName)

import SimpleITK as sitk
import tensorflow as tf
import numpy as np
import os
import json
from tensorflow.python.estimator import run_config as run_config_lib

#from azure.datalake.store import core, lib, multithread
from dltk.io.augmentation import extract_random_example_array, flip
from dltk.io.preprocessing import whitening


def read_fn(file_references, mode, params=None):
    """A custom python read function for interfacing with nii image files.

    Args:
        file_references (list): A list of lists containing file references,
            such as [['id_0', 'image_filename_0', target_value_0], ...,
            ['id_N', 'image_filename_N', target_value_N]].
        mode (str): One of the tf.estimator.ModeKeys strings: TRAIN, EVAL or
            PREDICT.
        params (dict, optional): A dictionary to parametrise read_fn outputs
            (e.g. reader_params = {'n_examples': 10, 'example_size':
            [64, 64, 64], 'extract_examples': True}, etc.).

    Yields:
        dict: A dictionary of reader outputs for dltk.io.abstract_reader.
    """
    print('Reading the dataset from Datalakestore (2mm NIfTI images)....')

    def _augment(img):
        """An image augmentation function"""
        return flip(img, axis=2)

    for f in file_references:
        subject_id = f[0]
        data_path = '/clusters/DLTK_IXI_Dataset/2mm'
        
        # Read the image nii with sitk
        ##t1_fn = os.path.join(data_path, '{}/T1_2mm.nii.gz'.format(subject_id))        
        ##t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(t1_fn)))
        t1_fn = os.path.join(data_path, '{}/T1_2mm.nii.gz'.format(subject_id))
        #with adlsFileSystemClient.open(t1_fn, 'rb') as f:
            # img = sitk.ReadImage(str(f))  
            # sitk::ERROR: The file "<ADL file: /clusters/DLTK_IXI_Dataset/2mm/IXI012/T1_2mm.nii.gz>" does not exist.
            # sitk seems only read from local path....how to read from remote path????????
            # for short term download to local path
            # rpath is datalakestore, lpath is local file path both have the same root structure '/clusters/DLTK_IXI_Dataset/'
        multithread.ADLDownloader(adlsFileSystemClient, rpath=t1_fn, lpath=t1_fn, nthreads=5, chunksize=2**24, overwrite=True)
        img = sitk.ReadImage(str(t1_fn))  
        # you need http://imagej.net/Fiji#Downloads app to show the img.  More discussion and instruction: https://stackoverflow.com/questions/45682319/simpleitk-show-generates-error-in-imagej-on-linux 
        ##sitk.Show(img)
        t1 = sitk.GetArrayFromImage(img) 

        # Normalise volume image
        t1 = whitening(t1)
        images = np.expand_dims(t1, axis=-1).astype(np.float32)
        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': images}, 'img_id': subject_id}
            
        # Parse the sex classes from the file_references [1,2] and shift them
        # to [0,1]
        sex = np.int(f[1]) - 1
        y = np.expand_dims(sex, axis=-1).astype(np.int32)
        
        # Augment if used in training mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            images = _augment(images)
        # Check if the reader is supposed to return training examples or full images
        if params['extract_examples']:
            images = extract_random_example_array(
                image_list=images,
                example_size=params['example_size'],
                n_examples=params['n_examples'])
            for e in range(params['n_examples']):
                yield {'features': {'x': images[e].astype(np.float32)},
                       'labels': {'y': y.astype(np.float32)},
                       'img_id': subject_id}
        else:
            yield {'features': {'x': images},
                   'labels': {'y': y.astype(np.float32)},
                   'img_id': subject_id}

    return

import pandas as pd

from dltk.networks.regression_classification.resnet import resnet_3d
from dltk.io.abstract_reader import Reader

##from reader import read_fn
EVAL_EVERY_N_STEPS = 100
EVAL_STEPS = 5

NUM_CLASSES = 2
NUM_CHANNELS = 1

BATCH_SIZE = 8
SHUFFLE_CACHE_SIZE = 32

MAX_STEPS = 50000

def model_fn(features, labels, mode, params):
    """Model function to construct a tf.estimator.EstimatorSpec. It creates a
        network given input features (e.g. from a dltk.io.abstract_reader) and
        training targets (labels). Further, loss, optimiser, evaluation ops and
        custom tensorboard summary ops can be added. For additional information,
         please refer to https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#model_fn.

    Args:
        features (tf.Tensor): Tensor of input features to train from. Required
            rank and dimensions are determined by the subsequent ops
            (i.e. the network).
        labels (tf.Tensor): Tensor of training targets or labels. Required rank
            and dimensions are determined by the network output.
        mode (str): One of the tf.estimator.ModeKeys: TRAIN, EVAL or PREDICT
        params (dict, optional): A dictionary to parameterise the model_fn
            (e.g. learning_rate)

    Returns:
        tf.estimator.EstimatorSpec: A custom EstimatorSpec for this experiment
    """

    # 1. create a model and its outputs
    net_output_ops = resnet_3d(
        features['x'],
        num_res_units=2,
        num_classes=NUM_CLASSES,
        filters=(16, 32, 64, 128, 256),
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        mode=mode,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # 1.1 Generate predictions only (for `ModeKeys.PREDICT`)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=net_output_ops,
            export_outputs={'out': tf.estimator.export.PredictOutput(net_output_ops)})

    # 2. set up a loss function
    one_hot_labels = tf.reshape(tf.one_hot(labels['y'], depth=NUM_CLASSES), [-1, NUM_CLASSES])

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=one_hot_labels,
        logits=net_output_ops['logits'])

    # 3. define a training op and ops for updating moving averages (i.e. for
    # batch normalisation)
    global_step = tf.train.get_global_step()
    optimiser = tf.train.AdamOptimizer(
        learning_rate=params["learning_rate"],
        epsilon=1e-5)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimiser.minimize(loss, global_step=global_step)

    # 4.1 (optional) create custom image summaries for tensorboard
    my_image_summaries = {}
    my_image_summaries['feat_t1'] = features['x'][0, 32, :, :, 0]

    expected_output_size = [1, 96, 96, 1]  # [B, W, H, C]
    [tf.summary.image(name, tf.reshape(image, expected_output_size))
     for name, image in my_image_summaries.items()]

    # 4.2 (optional) track the rmse (scaled back by 100, see reader.py)
    acc = tf.metrics.accuracy
    prec = tf.metrics.precision
    eval_metric_ops = {"accuracy": acc(labels['y'], net_output_ops['y_']),
                       "precision": prec(labels['y'], net_output_ops['y_'])}

    # 5. Return EstimatorSpec object
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=net_output_ops,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)

def train(_config):
    np.random.seed(42)
    tf.set_random_seed(42)

    print('Setting up....Task Type: ', _config._task_type)

    # Parse csv files for file names
    with adlsFileSystemClient.open('/clusters/DLTK_IXI_Dataset/demographic_HH.csv', 'rb') as f:
        all_filenames = pd.read_csv(f, dtype=object, keep_default_na=False, na_values=[]).as_matrix()
    train_filenames = all_filenames[:150]
    val_filenames = all_filenames[150:]

    # Set up a data reader to handle the file i/o.
    reader_params = {'n_examples': 2,
                     'example_size': [64, 96, 96],
                     'extract_examples': True}

    reader_example_shapes = {'features': {'x': reader_params['example_size'] + [NUM_CHANNELS]},
                             'labels': {'y': [1]}}
    reader = Reader(read_fn,
                    {'features': {'x': tf.float32},
                     'labels': {'y': tf.int32}})

    # Get input functions and queue initialisation hooks for training and
    # validation data
    train_input_fn, train_qinit_hook = reader.get_inputs(
        file_references=train_filenames,
        mode=tf.estimator.ModeKeys.TRAIN,
        example_shapes=reader_example_shapes,
        batch_size=BATCH_SIZE,
        shuffle_cache_size=SHUFFLE_CACHE_SIZE,
        params=reader_params)

    val_input_fn, val_qinit_hook = reader.get_inputs(
        file_references=val_filenames,
        mode=tf.estimator.ModeKeys.EVAL,
        example_shapes=reader_example_shapes,
        batch_size=BATCH_SIZE,
        shuffle_cache_size=SHUFFLE_CACHE_SIZE,
        params=reader_params)

    # Instantiate the neural network estimator
    nn = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_path,
        params={"learning_rate": 0.001},
        config=_config)
        #config=tf.estimator.RunConfig())

    # Hooks for validation summaries
    val_summary_hook = tf.contrib.training.SummaryAtEndHook(
        os.path.join(model_path, 'eval'))
    step_cnt_hook = tf.train.StepCounterHook(every_n_steps=EVAL_EVERY_N_STEPS,
                                             output_dir=model_path)

    print('Starting training...')
    try:
        for _ in range(MAX_STEPS // EVAL_EVERY_N_STEPS):
            nn.train(
                input_fn=train_input_fn,
                hooks=[train_qinit_hook, step_cnt_hook],
                steps=EVAL_EVERY_N_STEPS)

            ##if args.run_validation:
            results_val = nn.evaluate(
                input_fn=val_input_fn,
                hooks=[val_qinit_hook, val_summary_hook],
                steps=EVAL_STEPS)
            print('Step = {}; val loss = {:.5f};'.format(
                results_val['global_step'],
                results_val['loss']))

    except KeyboardInterrupt:
        pass

    # When exporting we set the expected input shape to be arbitrary.
    export_dir = nn.export_savedmodel(
        export_dir_base=model_path,
        serving_input_receiver_fn=reader.serving_input_receiver_fn(
            {'features': {'x': [None, None, None, NUM_CHANNELS]},
             'labels': {'y': [1]}}))
    print('Model saved to {}.'.format(export_dir))
    multithread.ADLUploader(adlsFileSystemClient, lpath=export_dir, rpath=export_dir, nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)


# Set up argument parser
flags = tf.app.flags
#flags.DEFINE_string("task_index", None, "Worker task index, should be >= 0. task_index=0 is " "the master worker task the performs the variable " "initialization ")
flags.DEFINE_integer("task_index", 0, "Worker task index, should be >= 0. task_index=0 is " "the master worker task the performs the variable " "initialization ")
flags.DEFINE_integer("num_gpus", 1, "Total number of gpus for each machine." "If you don't use GPU, please set it to '0'")
flags.DEFINE_string("ps_hosts","localhost:2222", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None,"job name: worker or ps")
flags.DEFINE_string("cuda_devices", None,"cuda_devices: 0")
FLAGS = flags.FLAGS

#Since I can't define the task type from the BatchAI env, so I've to set it manually
ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")
#Set this use case with 3 nodes, 1 dedicated Chief also act as PS and 2 worker node (take the 1st node of worker_hosts as chief)
##print("Original - ps: ", ps_hosts, "worker: ", worker_hosts)
chief_node = worker_hosts[0]
chief_node_list = []
chief_node_list.append(chief_node)
del worker_hosts[0]

#Adjust the worker index since I reserved the 1 worker as Chief node
if FLAGS.job_name == "worker":
    task_index = FLAGS.task_index -1
else:
    task_index = FLAGS.task_index

os.environ['TF_CONFIG'] = json.dumps({
    ##'cluster': cluster,
    'cluster': {
        "chief" : chief_node_list,
        "ps": ps_hosts,
        "worker": worker_hosts
    },
    'task' : {
        'type' : FLAGS.job_name,
        'index': task_index,
    }
})
print('script starting.....TF_CONFIG: ', os.environ['TF_CONFIG'])
# Note: Then tf.estimator.RunConfig() should pick up these cluster spec
config = tf.estimator.RunConfig()

#Checking........
if FLAGS.job_name == "chief":
    assert config._task_type == run_config_lib.TaskType.CHIEF
    #print(config.cluster_spec.as_dict().get(run_config_lib.TaskType.CHIEF, []))
elif FLAGS.job_name == "worker":
    assert config._task_type == run_config_lib.TaskType.WORKER
    #print(config.cluster_spec.as_dict().get(run_config_lib.TaskType.WORKER, []))
elif FLAGS.job_name == "ps":
    assert config._task_type == run_config_lib.TaskType.PS
    #print(config.cluster_spec.as_dict().get(run_config_lib.TaskType.PS, []))

#if FLAGS.job_name not in config.cluster_spec.jobs:
if config._task_type not in config.cluster_spec.jobs:
    print('Task Type: ', config._task_type, ' is in not the cluster_spec.jobs!!')
    raise ValueError('taskType', config._task_type, 'spec', config.cluster_spec)
else:
    print('Task Type: ', config._task_type, ' is in the cluster_spec.jobs')
print(config.cluster_spec.as_dict().get(config._task_type, []))

#Total number of chief and worker nodes (note: I share the ps on chief node)
print(len(config.cluster_spec.as_dict().get(run_config_lib.TaskType.WORKER, [])) +  len(config.cluster_spec.as_dict().get(run_config_lib.TaskType.CHIEF, [])))

# GPU allocation options (see if BatchAI cmd line can take care this)
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_devices

model_path = '/clusters/DLTK_IXI_Dataset/IXI_sex_classification/'
adlsFileSystemClient.mkdir(path=model_path)
train(config)
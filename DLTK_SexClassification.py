## Required for Azure Data Lake Store filesystem management
from azure.datalake.store import core, lib, multithread
##--Option1(SignIn): User Acct, interactively sign in to AzDataLake Store, storebotdatalakestore
#
##subscriptionId = '851da8fc-5b5f-48f2-9e14-395ce8ace4bf'
adlsAccountName = 'storebotdatalakestore'
adlCreds = lib.auth(tenant_id = '4dced229-4c95-476d-b76b-34d306d723eb', resource = 'https://datalake.azure.net/')

##--Create a filesystem client object
adlsFileSystemClient = core.AzureDLFileSystem(adlCreds, store_name=adlsAccountName)

##--create local path that mimic remote path for operation
#model_path = '/clusters/DLTK_IXI_Dataset/IXI_sex_classification/'
# Make sure that the RunConfig.model_dir for all workers is set to the same shared directory -> https://github.com/Azure/BatchAI/blob/master/documentation/using-azure-cli-20.md
model_path ='/mnt/batch/tasks/shared/LS_root/mounts/external/NFS'
data_path = '/clusters/DLTK_IXI_Dataset/2mm/'

import SimpleITK as sitk
import tensorflow as tf
import numpy as np
import os
import json
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.contrib.learn import learn_runner
#import azure.mgmt.batchai.models as models  #pip install azure-mgmt-batchai

# Show debugging output
tf.logging.set_verbosity(tf.logging.DEBUG)  #generate too much log, too large more than 1M, can't viwe at batchai
#tf.logging.set_verbosity(tf.logging.INFO)  #generate too much log, too large more than 1M, can't viwe at batchai

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

    image_array = []
    label_array = []
    for f in file_references:
        subject_id = f[0]

        # Read the image nii with sitk
        ##t1_fn = os.path.join(data_path, '{}/T1_2mm.nii.gz'.format(subject_id))
        ##t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(t1_fn)))
        t1_fn = os.path.join(data_path, '{}/T1_2mm.nii.gz'.format(subject_id))
        print(t1_fn)
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
            print('read_fn Predict')

        # Parse the sex classes from the file_references [1,2] and shift them
        # to [0,1]
        sex = np.int(f[1]) - 1
        y = np.expand_dims(sex, axis=-1).astype(np.int32)

        # Augment if used in training mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            images = _augment(images)
            print('read_fn Train')
        # Check if the reader is supposed to return training examples or full images
        if params['extract_examples']:
            #print('read_fn params extract_examples')
            images = extract_random_example_array(
                image_list=images,
                example_size=params['example_size'],
                n_examples=params['n_examples'])
            for e in range(params['n_examples']):
                #print ('e: ', e)
##                yield {'features': {'x': images[e].astype(np.float32)},
##                      'labels': {'y': y.astype(np.float32)},
##                       'img_id': subject_id}
                image_array.append(images[e].astype(np.float32))
                label_array.append(y.astype(np.int32))
        else:
            print('read_fn params yield last')
##            yield {'features': {'x': images},
##                   'labels': {'y': y.astype(np.float32)},
##                   'img_id': subject_id}
            image_array.append(images)
            label_array.append(y.astype(np.int32))

    print("read_fn yield output_array with image shape = ", images.shape, "label shape = ", y.shape)
    yield {'x': np.array(image_array), 'y': np.array(label_array)}

import pandas as pd

#Ref: https://dltk.github.io/DLTK/_modules/dltk/networks/regression_classification/resnet.html
from dltk.networks.regression_classification.resnet import resnet_3d
from dltk.io.abstract_reader import Reader

##from reader import read_fn
EVAL_EVERY_N_STEPS = 100
EVAL_STEPS = 5

NUM_CLASSES = 2
NUM_CHANNELS = 1

BATCH_SIZE = 8
SHUFFLE_CACHE_SIZE = 32

#MAX_STEPS = 50000  #Note: use MAX_STEPS to ensure not run more than the steps originally in cluster env.  Num of Step = Num of epoch x Num of Train Samples x (1/Batch Size).  To avoid Num og Worker run independently with its own epoch/iteration, use TF data queue!
# use lower /max step for quicker demo
MAX_STEPS = 10000

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
    print('Estimator constructing model function.....')

    # 1. create a model and its outputs
    net_output_ops = resnet_3d(
        inputs=features,
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
##    one_hot_labels = tf.reshape(tf.one_hot(labels['y'], depth=NUM_CLASSES), [-1, NUM_CLASSES])
    one_hot_labels = tf.reshape(tf.one_hot(labels, depth=NUM_CLASSES), [-1, NUM_CLASSES])

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
##    my_image_summaries['feat_t1'] = features['x'][0, 32, :, :, 0]
    my_image_summaries['feat_t1'] = features[0, 32, :, :, 0]

    expected_output_size = [1, 96, 96, 1]  # [B, W, H, C]
    [tf.summary.image(name, tf.reshape(image, expected_output_size))
     for name, image in my_image_summaries.items()]

    # 4.2 (optional) track the rmse (scaled back by 100, see reader.py)
    acc = tf.metrics.accuracy
    prec = tf.metrics.precision
    eval_metric_ops = {"accuracy": acc(labels, net_output_ops['y_']),
                       "precision": prec(labels, net_output_ops['y_'])}

    print('Returning Estimator Spec Object......')
    #Ref: https://github.com/tensorflow/tensorflow/issues/14042
    summary_hook = tf.train.SummarySaverHook(
        save_steps=EVAL_STEPS,
        output_dir=model_path,
        summary_op=tf.summary.merge_all())
    # 5. Return EstimatorSpec object
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=net_output_ops,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      training_hooks=[summary_hook])

# Define data loaders
class IteratorInitializerHook(tf.train.SessionRunHook):

    """Hook to initialise data iterator after Session is created."""
    def __init__(self):
        """
        if config._task_type == run_config_lib.TaskType.CHIEF:
            print("interator session create init global var on..........",  config._task_type)
            tf.global_variables_initializer()
        else:
            print("interator session create do nothing on var init.........",  config._task_type)
        """
        print("interator session create..........", config._task_type)
        #tf.global_variables_initializer()
        #tf.local_variables_initializer()
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        print("interator session after created..........",  config._task_type)
        self.iterator_initializer_func(session)

def simple_setter(ps_device="/job:ps/task:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == "Variable":
            return ps_device
        else:
            return "/job:worker/task:%d" % (FLAGS.task)
    return _assign

def get_train_inputs(batch_size, file_reference, mode, params, dtypes, example_shapes, shuffle_cache_size):

    iterator_initializer_hook = IteratorInitializerHook()
    print('get_train_inputs....')
    def train_inputs():
        g = read_fn(file_reference, mode, params)
        g = next(g)
        images = g['x']
        labels = g['y']
##        images = images.reshape([1, 96, 96, 1]) #I see error, cannot reshape array of size 589824 into shape (1,96,96,1) and printing the image shape from the read_fn is rank of 5 for 3d_resnet B, 3d?, H, W and C
##        images_placeholder = tf.placeholder(tf.float32, shape=images.shape)
        images_placeholder = tf.placeholder(tf.float32, shape=[300, 64, 96, 96, 1]) #batch size 300 from error message, better use np.expand_dims(img, axis=0) to add the batch dimension, keep the input images itseld as 64(3d), 96(H), 96(W), 1(Channel)
        #OR
        #images_placeholder = tf.placeholder(tf.float32, shape=np.expand_dims(example_shapes, axis=0)) #auto add batch size to example_shpaes = 64(nii format?), 96(W), 96(H), 1(C) and the batch size shoule be 300 (images) in runtime
##        labels_placeholder = tf.placeholder(tf.int32, shape=labels.shape)
        labels_placeholder = tf.placeholder(tf.int32, shape=[300, 1])
        #OR
        #labels_placeholder = tf.placeholder(tf.int32, shape=np.expand_dims(1, axis=0))
        # Build dataset iterator
        ##dataset = tf.contrib.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))  ##for older than tf1.4
        dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
        dataset = dataset.repeat(None)  # Infinite iterations
        dataset = dataset.shuffle(shuffle_cache_size)
        dataset = dataset.batch(batch_size)
        #tf.session().run([tf.global_variables_initializer(dataset), tf.local_variables_initializer()])
        tf.global_variables_initializer(dataset)
        tf.local_variables_initializer()
        ##iterator = dataset.make_initializable_iterator()
        iterator = dataset.make_one_shot_iterator()
        next_example, next_label = iterator.get_next()
#        with tf.device("/job:chief/task:0"):
##        next_dict = iterator.get_next()
        #with tf.device(simple_setter):
        ##next_example, next_label = iterator.get_next()
        # Set runhook to initialize iterator
        print('before sess run....')
#       iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer)
        iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer, feed_dict={images_placeholder: images, labels_placeholder: labels})
        # Return batched (features, labels)
#       return next_dict['features'], next_dict.get('labels')
##      return next_dict[0], next_dict[1]
        return next_example, next_label
    # Return function and hook
    return train_inputs, iterator_initializer_hook

def my_input_fn(batch_size, file_reference, mode, params, dtypes, example_shapes, shuffle_cache_size):

    #Note: Estimator will automatically create and initialize an initializable iterator when you return a tf.data.Dataset from your input_fn. This enables you to write the following code, without having to worry about initialization or hooks https://stackoverflow.com/questions/48614529/how-to-use-the-iterator-make-initializable-iterator-of-tensorflow-within-a-in
    ##iterator_initializer_hook = IteratorInitializerHook()
    print('my_input_fn....')
    g = read_fn(file_reference, mode, params)
    g = next(g)
    images = g['x']
    labels = g['y']
    images_placeholder = tf.placeholder(np.float32, shape=[300, 64, 96, 96,
                                                           1])  # batch size 300 from error message, better use np.expand_dims(img, axis=0) to add the batch dimension, keep the input images itseld as 64(3d), 96(H), 96(W), 1(Channel)
    labels_placeholder = tf.placeholder(np.float32, shape=[300, 1])
    #dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
    # new code to replace placeholder
    images = images.reshape([300, 64, 96, 96, 1])
    labels = labels.reshape([300,1])
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.repeat(None)  # Infinite iterations
    if shuffle_cache_size > 0:
        dataset = dataset.shuffle(shuffle_cache_size)
        print("my_input_fn ->train_input call")
    else:
        print("my_input_fn ->eval_input call")
    dataset = dataset.batch(batch_size)
    ##tf.global_variables_initializer()
    ##tf.local_variables_initializer()
    ##iterator = dataset.make_initializable_iterator()
    #iterator = dataset.make_one_shot_iterator()
    ##tf.Session().run(iterator.initializer)
    #next_example, next_label = iterator.get_next()
    ##next_example = iterator.get_next()
    ##iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer, feed_dict={images_placeholder: images, labels_placeholder: labels})

    # https://github.com/google/seq2seq/issues/230 See if this can help to fix the report_uninitialized_variables on Worker nodes
    """
    with tf.Session() as sess:
        sess.run(iterator.initializer,
                 feed_dict={images_placeholder: images,
                            labels_placeholder: labels})
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    """
    #return next_example, next_label
    return dataset

def train(_config):
    np.random.seed(42)
    tf.set_random_seed(42)

    print('Setting up....Task Type: ', _config._task_type)

    # Parse csv files for file names
    with adlsFileSystemClient.open('/clusters/DLTK_IXI_Dataset/demographic_HH.csv', 'rb') as f:
        all_filenames = pd.read_csv(f, dtype=object, keep_default_na=False, na_values=[]).as_matrix()
    train_filenames = all_filenames[:150]  #1st 150 images
    eval_filenames = all_filenames[150:]    # then another 150+ images

    # Set up a data reader to handle the file i/o.
    reader_params = {'n_examples': 2,
                     'example_size': [64, 96, 96],
                     'extract_examples': True}

    reader_example_shapes = {'features': {'x': reader_params['example_size'] + [NUM_CHANNELS]},
                             'labels': {'y': [1]}}

    #reader = Reader(read_fn, {'features': {'x': tf.float32}, 'labels': {'y': tf.int32}})

    # Get input functions and queue initialisation hooks for training and
    """
    # Note: drop this old way from using tf.contrib.learn https://towardsdatascience.com/how-to-move-from-tf-contrib-learn-estimator-to-core-tensorflow-tf-estimator-af07b2d21f34
    train_input_fn, train_qinit_hook = get_train_inputs(
        batch_size=BATCH_SIZE,
        file_reference=train_filenames,
        mode=tf.estimator.ModeKeys.TRAIN,
        params=reader_params,
        dtypes={'features': {'x': tf.float32},'labels': {'y': tf.int32}},
        example_shapes=reader_example_shapes,
        shuffle_cache_size=SHUFFLE_CACHE_SIZE)
    
    eval_input_fn, eval_qinit_hook = get_train_inputs(
        file_reference=eval_filenames,
        mode=tf.estimator.ModeKeys.EVAL,
        example_shapes=reader_example_shapes,
        batch_size=BATCH_SIZE,
        shuffle_cache_size=SHUFFLE_CACHE_SIZE,
        params=reader_params)
    
    eval_input_fn, eval_qinit_hook = get_train_inputs(
        batch_size=BATCH_SIZE,
        file_reference=eval_filenames,
        mode=tf.estimator.ModeKeys.EVAL,
        params=reader_params,
        dtypes={'features': {'x': tf.float32}, 'labels': {'y': tf.int32}},
        example_shapes=reader_example_shapes,
        shuffle_cache_size=SHUFFLE_CACHE_SIZE)
    """
    #tf.estimator.inputs.numpy_input_fn expects numpy array,feature_batch and label_batch are dict, so need to convert to numpy array
##    feature_batch, label_batch = train_input_fn()

##    my_train_input_fn = tf.estimator.inputs.numpy_input_fn(
##        x={'x': np.array(list(feature_batch.items()))},
##        y=np.array(list(label_batch.items())),
##        num_epochs=1,
##        shuffle=True
##    )


    ##val_input_fn, val_qinit_hook = reader.get_inputs(
    ##    file_references=val_filenames,
    ##    mode=tf.estimator.ModeKeys.EVAL,
    ##    example_shapes=reader_example_shapes,
    ##    batch_size=BATCH_SIZE,
    ##    shuffle_cache_size=SHUFFLE_CACHE_SIZE,
    ##    params=reader_params)

    # Instantiate the neural network estimator
    # Note: tf.estimator.Estimator, which actually uses replica_device_setter https://stackoverflow.com/questions/46974717/tf-train-replica-device-setter-needed-with-tf-contrib-learn-experiment?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    #       So I've this error https://github.com/tensorflow/tensorflow/issues/7785 seems the device setter is not correct which means my tf is running local but estimater expect in network?
    nn = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_path,
        params={"learning_rate": 0.001},
        config=_config)
        #config=tf.estimator.RunConfig())

    # Hooks for validation summaries
    ##val_summary_hook = tf.contrib.training.SummaryAtEndHook(os.path.join(model_path, 'eval'))
    ##step_cnt_hook = tf.train.StepCounterHook(every_n_steps=EVAL_EVERY_N_STEPS, output_dir=model_path)

    print('Starting training...')
    train_input = lambda: my_input_fn(
        batch_size=BATCH_SIZE,
        file_reference=train_filenames,
        mode=tf.estimator.ModeKeys.TRAIN,
        params=reader_params,
        dtypes={'features': {'x': tf.float32},'labels': {'y': tf.int32}},
        example_shapes=reader_example_shapes,
        shuffle_cache_size=SHUFFLE_CACHE_SIZE)


    train_spec = tf.estimator.TrainSpec(train_input, max_steps=MAX_STEPS)
    print('train_spec return....')

    eval_input = lambda: my_input_fn(
        batch_size=BATCH_SIZE,
        file_reference=eval_filenames,
        mode=tf.estimator.ModeKeys.TRAIN,
        params=reader_params,
        dtypes={'features': {'x': tf.float32}, 'labels': {'y': tf.int32}},
        example_shapes=reader_example_shapes,
        shuffle_cache_size=0)

    eval_spec = tf.estimator.EvalSpec(eval_input)
    print('eval_spec return....')

    tf.estimator.train_and_evaluate(nn, train_spec, eval_spec)
    print('train_and_evaluate return....')

    """
    # Note: drop this old way from using tf.contrib.learn https://towardsdatascience.com/how-to-move-from-tf-contrib-learn-estimator-to-core-tensorflow-tf-estimator-af07b2d21f34
    # Ref: https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0
    experiment = tf.contrib.learn.Experiment(
        estimator=nn,  # Estimator
        train_input_fn=train_input_fn,  # First-class function
        eval_input_fn=eval_input_fn,  # First-class function
        train_steps=MAX_STEPS,  #...steps
        min_eval_frequency=EVAL_EVERY_N_STEPS,  # Eval frequency
        train_monitors=[train_qinit_hook],  # Hooks for training
        eval_hooks=[eval_qinit_hook],  # Hooks for evaluation
        eval_steps=EVAL_STEPS  # Use evaluation feeder until its empty
    )
    params = tf.contrib.training.HParams(
        learning_rate=0.001,
        n_classes=NUM_CLASSES,
        train_steps=MAX_STEPS,
        min_eval_frequency=EVAL_EVERY_N_STEPS
    )
    learn_runner.run(
        experiment_fn=experiment,  # First-class function
        run_config=config,  # RunConfig
        schedule="train_and_evaluate",  # What to run
        hparams=params  # HParams
    )
    """

    # nn.train(input_fn=train_input_fn, max_steps=MAX_STEPS, hooks=train_qinit_hook)
    ##OR below from original dltk example code
    """ 
    try:
        for _ in range(MAX_STEPS // EVAL_EVERY_N_STEPS):
            nn.train(
                input_fn=train_input_fn,
                ##input_fn=train_input,
                hooks=[train_qinit_hook, step_cnt_hook],
                ##hooks=[iterator_initializer_hook, step_cnt_hook],
                steps=EVAL_EVERY_N_STEPS)

            ##if args.run_validation: Temporary as I've yet figure how to set the Evaluator spec on BatchAI https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig
            ##results_val = nn.evaluate(
            ##    input_fn=eval_input_fn,
            ##    hooks=[val_qinit_hook, val_summary_hook],
            ##   steps=EVAL_STEPS)
            ##print('Step = {}; val loss = {:.5f};'.format(
            ##    results_val['global_step'],
            ##    results_val['loss']))
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
    """

# Set up argument parser
flags = tf.app.flags
flags.DEFINE_integer("task_index", 0, "Worker task index, should be >= 0. task_index=0 is " "the master worker task the performs the variable " "initialization ")
flags.DEFINE_integer("num_gpus", 1, "Total number of gpus for each machine." "If you don't use GPU, please set it to '0'")
flags.DEFINE_string("ps_hosts","localhost:2222", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None,"job name: worker or ps")
flags.DEFINE_string("cuda_devices", None,"cuda_devices: 0")
FLAGS = flags.FLAGS

# GPU allocation options (see if BatchAI cmd line can take care this)
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_devices

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

if FLAGS.job_name == 'chief':
    task_type = run_config_lib.TaskType.CHIEF
elif FLAGS.job_name == 'worker':
    task_type = run_config_lib.TaskType.WORKER
elif FLAGS.job_name == 'ps':
    task_type = run_config_lib.TaskType.PS
else:
    print("ERROR TYPE! is either chief, worker or ps")

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        "chief": chief_node_list,
        "ps": ps_hosts,
        "worker": worker_hosts
    },
    'task': {
        'type': FLAGS.job_name,
        'index': task_index,
    },
    'environment': 'cloud'
})
print('script starting.....TF_CONFIG: ', os.environ['TF_CONFIG'])

# Note: Then tf.estimator.RunConfig() should pick up these cluster spec
if FLAGS.job_name == 'ps':
    config = tf.estimator.RunConfig()
else:
    # trains faster with GPU for this model.                                                                        #using gpu only, 0 using cpu only
    #config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(log_device_placement=True, device_count={'GPU': 1}, device_filters=['/job:ps', '/job:worker/task:%d' % task_index]))
    #config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(log_device_placement=True, device_count={'GPU': 1}))
    custom_config = tf.ConfigProto(allow_soft_placement=True,
                                   log_device_placement=True,
                                   gpu_options=tf.GPUOptions(force_gpu_compatible=True)
                                   )#,
                                   #device_filters=['/job:ps', '/job:worker/task:%d' % task_index])
    #custom_config.gpu_options.allocator_type = 'BFC'
    custom_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(session_config=custom_config)

# Ref: https://cloud.google.com/blog/big-data/2018/02/easy-distributed-training-with-tensorflow-using-tfestimatortrain-and-evaluate-on-cloud-ml-engine
config = config.replace(model_dir=model_path)

# Ref: https://stackoverflow.com/questions/43084960/tensorflow-variables-are-not-initialized-using-between-graph-replication
# remove the interdependency between the workers, by setting a "device filter" to see if it can fix the dltk-job1 error below, still can't fix...??????
#config_proto = tf.ConfigProto(device_filters=['/job:ps', '/job:worker/task:%d' % task_index])
#config = tf.estimator.RunConfig(session_config=config_proto)

# this is old way distributed TF
#server = tf.train.Server(config.cluster_spec, job_name=FLAGS.job_name, task_index=task_index)

#Checking........
if FLAGS.job_name == "chief":
    assert config._task_type == run_config_lib.TaskType.CHIEF
    #print(config.cluster_spec.as_dict().get(run_config_lib.TaskType.CHIEF, []))
    #with tf.train.MonitoredTrainingSession(server.target) as sess:
    #    while not sess.should_stop():
    #        sess.run(tf.global_variables_initializer())
            #train(config)
elif FLAGS.job_name == "worker":
    assert config._task_type == run_config_lib.TaskType.WORKER
    #print(config.cluster_spec.as_dict().get(run_config_lib.TaskType.WORKER, []))
    #with tf.train.MonitoredTrainingSession(server.target) as sess:
    #    while not sess.should_stop():
    #        sess.run(tf.global_variables_initializer())
            #train(config)
elif FLAGS.job_name == "ps":
    assert config._task_type == run_config_lib.TaskType.PS
    #print(config.cluster_spec.as_dict().get(run_config_lib.TaskType.PS, []))
    #server.join()

#if FLAGS.job_name not in config.cluster_spec.jobs:
if config._task_type not in config.cluster_spec.jobs:
    print('Task Type: ', config._task_type, ' is in not the cluster_spec.jobs!!')
    raise ValueError('taskType', config._task_type, 'spec', config.cluster_spec)
else:
    print('Task Type: ', config._task_type, ' is in the cluster_spec.jobs')
print(config.cluster_spec.as_dict().get(config._task_type, []))

#Total number of chief and worker nodes (note: I share the ps on chief node)
print(len(config.cluster_spec.as_dict().get(run_config_lib.TaskType.WORKER, [])) +  len(config.cluster_spec.as_dict().get(run_config_lib.TaskType.CHIEF, [])))

#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer()) #refer result of dltk-job1: error in worker1,2 -> Ready_for_local_init_op:  Variables not initialized: global_step, conv3d/kernel

# Construct device setter object
device_setter = tf.train.replica_device_setter(cluster=config.cluster_spec)
with tf.device(device_setter):
    train(config)
"""
if FLAGS.job_name == "ps":
    train(config)
else:
    with tf.device('/gpu:0'):
        train(config)
"""
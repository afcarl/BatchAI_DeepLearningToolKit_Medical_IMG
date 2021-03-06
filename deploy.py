# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.contrib import predictor

from dltk.io.augmentation import extract_random_example_array

from reader import read_fn

READER_PARAMS = {'extract_examples': False}
N_VALIDATION_SUBJECTS = 178 #read the whole csv rows. original is 28  


def predict(args):
    # Read in the csv with the file names you would want to predict on

    file_names = pd.read_csv(
        args.csv,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()


    # We trained on the first 4 subjects, so we predict on the rest
    file_names = file_names[-N_VALIDATION_SUBJECTS:] #note:file_names[-28:] is the last 28 rows which is startiing from row-0 is IXI566 to row-27 IXI646
    #OR
    #file_names = file_names[156:] #note: this is IXI586, male
    # From the model_path, parse the latest saved model and restore a
    # predictor from it
    export_dir = \
        [os.path.join(args.model_path, o) for o in sorted(os.listdir(args.model_path))
         if os.path.isdir(os.path.join(args.model_path, o)) and o.isdigit()][-1]

    print('Loading from {}'.format(export_dir))
    my_predictor = predictor.from_saved_model(export_dir)

    # Iterate through the files, predict on the full volumes and compute a Dice
    # coefficient
    accuracy = []
    for output in read_fn(file_references=file_names,
                          mode=tf.estimator.ModeKeys.EVAL,
                          input_img=args.img,
                          params=READER_PARAMS):
        t0 = time.time()
        # Parse the read function output and add a dummy batch dimension as
        # required
        img = output['features']['x']
        lbl = output['labels']['y']
        test_id = output['img_id']

        # We know, that the training input shape of [64, 96, 96] will work with
        # our model strides, so we collect several crops of the test image and
        # average the predictions. Alternatively, we could pad or crop the input
        # to any shape that is compatible with the resolution scales of the
        # model:

        num_crop_predictions = 6
        crop_batch = extract_random_example_array(
            image_list=img,
            example_size=[64, 96, 96],
            n_examples=num_crop_predictions)

        y_ = my_predictor.session.run(
            fetches=my_predictor._fetch_tensors['y_prob'],
            feed_dict={my_predictor._feed_tensors['x']: crop_batch})

        # Average the predictions on the cropped test inputs:
        y_ = np.mean(y_, axis=0)
        predicted_class = np.argmax(y_)

        if predicted_class == 0:
            predicted_class_result='Male'
        else:
            predicted_class_result='Female'

        if lbl == 0:
            lbl_is = 'Male'
        else:
            lbl_is = 'Female'
        # Calculate the accuracy for this subject
        accuracy.append(predicted_class == lbl)

        # Print outputs
        print('id={}; pred={}; true={}; run time={:0.2f} s; '
              ''.format(test_id, predicted_class_result, lbl_is, time.time() - t0))
              #''.format(test_id, predicted_class, lbl[0], time.time() - t0))
    print('accuracy={}'.format(np.mean(accuracy)))


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='IXI HH example sex classification deploy script')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='1')

    #parser.add_argument('--model_path', '-p', default='/tmp/IXI_sex_classification/')
    parser.add_argument('--model_path', '-p', default='IXI_HH_DCGAN_checkpoint_folder/IXI_HH_sex_classification')

    #parser.add_argument('--csv', default='../../../data/IXI_HH/demographic_HH.csv')
    parser.add_argument('--csv', default='DLTK_IXI_Dataset/Data/demographic_HH.csv')

    parser.add_argument('--img', default='IXI586')

    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Call training
    predict(args)
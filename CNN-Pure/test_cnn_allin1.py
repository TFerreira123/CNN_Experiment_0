import sys
sys.path.append('cnn_optimization/Utils')
import pickle
import time
import random
import os
from os import environ, listdir, makedirs
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
import json
#from keras.layers import Dense, Flatten, Dropout, BatchNormalization
#from keras.layers import Dense, Flatten, Dropout, BatchNormalization
#from keras.layers.convolutional import Conv2D, MaxPooling2D

#this changed in a recent version of keras, to this:
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D

from keras.models import Sequential
#from keras.utils import np_utils
#more changes:
from tensorflow.keras.utils import to_categorical
 
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score,accuracy_score 
import warnings
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
import keras_tuner

from History import History
from MyCallbacks import MyCallbacks
from keras.layers import Dense, concatenate, Input, LSTM, TimeDistributed, Bidirectional
from keras.layers import Flatten, Dropout, BatchNormalization
#from keras.layers.convolutional import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
 



warnings.simplefilter("ignore")

#CURR_PATH = Path('.').absolute().parent.__str__() + '/'
CURR_PATH = 'CNN-Pure/'

# Path to the directory with all necessary data
PATH_TO_DATASET = ''
# Results path based on evaluation strategy
EVAL_STRATEGY_PATH = 'CV_10_Fold_10_Reps/'
#EVAL_STRATEGY_PATH = 'TVT_70_15_15/'
#EVAL_STRATEGY_PATH = 'TVT_40_30_30/'
# Path to tvt splits
TVT_70_15_15_PATH = 'tvt_dataframes/tvt_701515/'
TVT_40_30_30_PATH = 'tvt_dataframes/tvt_403030/'
# Location of the .wav files converted from .mp3 as the intermediate step to generate Mel-spectrograms
WAV_LOCATION = 'total_wav_test/'
# Name of the file with the song ids and respective quadrants
CSV_NAME = 'CNNTestSet/merge_audio_complete_test_targets.csv'
# Name of the file containing all samples. The program expects a pickle file
# with the structure: (num_samples, (song_id, waveform, target))
SAMPLES_FILE = 'CNNTestSet/merge_audio_complete_test_normalized_16kHz'
# Name of the file containing all Mel-spectrogram representations of the samples.
# The program expects a pickle file with the structure: (num_of_samples, (width, height))
MEL_FILE_TEST = 'CNNTestSet/merge_audio_complete_test_16kHz_melspect_norm'
MEL_FILE_TRAIN = 'CNNTestSet/merge_audio_complete_16kHz_melspect_norm'
# Function that collects all the file names of the songs and sorts them by Song_ID and returns
# them as an array
# Path related with optimization strategy
PATH_TO_HP_SEARCH_FOLDS = 'folds/categorical_baseline_MERGE_Audio_Complete'
PATH_TO_TRIALS_SUMMARY = 'bayesian_tuner_summary/bayesian_cv_summary'
NAME_MUSIC = 'merge_audio_complete_test_16kHz_melspect_1_5s'

#################################
### Optimization Options ########
#################################
TRAIN_ACCURACY_TRESHOLD = 0.9
NO_IMPROVEMENT_OVER_EPOCHS = 15

#############################
### Utils ###################
#############################

REAL_FOLDER = 'real_annot_allin1'
PRED_FOLDER = 'pred_annot_allin1'
COMPARISON_SEG_OUTPUT = 'real_pred_seg_correction_allin1'
COMPARISON_SEG_OUTPUT_ALL = 'real_pred_seg_correction_allin1_all_songs/all_songs.txt'

# Flag for model to be built as unidirectional (False) or bidirectional (True)
bidirectional = False

def create_dirs_for_model_test(candidate, cv_eval, eval_all_trials=False):
    if eval_all_trials:
        path_for_results = EVAL_STRATEGY_PATH + candidate[3]
    else:
        path_for_results = EVAL_STRATEGY_PATH

    try:
        makedirs(path_for_results + 'conf_matrix/bs_{}_opt_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass

    try:
        makedirs(path_for_results + 'f1_macro_best100/bs_{}_opt_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass
    try:
        makedirs(path_for_results + 'precision/bs_{}_opt_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass
    try:
        makedirs(path_for_results + 'recall/bs_{}_opt_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass

    try:
        makedirs(path_for_results + 'f1_each/bs_{}_opt_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass
    try:
        makedirs(path_for_results + 'precision_each/bs_{}_opt_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass
    try:
        makedirs(path_for_results + 'recall_each/bs_{}_opt_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass

    try:
        makedirs(path_for_results + 'true_and_pred/bs_{}_opt_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass
    try:
        makedirs(path_for_results + 'scores/bs_{}_opt_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass

    return

# Define Custom Masking Layer
class CustomMasking(tf.keras.layers.Layer):
    def __init__(self, mask_value, **kwargs):
        super(CustomMasking, self).__init__(**kwargs)
        self.mask_value = mask_value

    def build(self, input_shape):
        super(CustomMasking, self).build(input_shape)

    def call(self, inputs, mask=None):
        mask = tf.not_equal(inputs, self.mask_value)
        return tf.where(mask, inputs, tf.zeros_like(inputs))

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, self.mask_value)

    def get_config(self):
        config = super(CustomMasking, self).get_config()
        config.update({"mask_value": self.mask_value})
        return config

def create_standard_CNN(name='standard', lr=0.0001, show_summary=False):
    # 16kHz
    # FIXME: changed to 2 seconds
    width_time = 48  # 16kHz
    num_freq_bins = 128
    num_classes = 4
    model = Sequential()
    filters = (3, 3)

    model.add(Conv2D(16, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                          kernel_initializer='he_normal', input_shape=(width_time, num_freq_bins, 1), name='conv_1'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', name='mp_1'))
    model.add(BatchNormalization(name='batch_norm_1'))
    model.add(Dropout(0.1, name='drop_1'))

    model.add(Conv2D(16, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                          kernel_initializer='he_normal', name='conv_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', name='mp_2'))
    model.add(BatchNormalization(name='batch_norm_2'))
    model.add(Dropout(0.1, name='drop_2'))

    model.add(Conv2D(16, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                          kernel_initializer='he_normal', name='conv_3'))
    model.add(MaxPooling2D(pool_size=(3, 2), padding='same', name='mp_3'))
    model.add(BatchNormalization(name='batch_norm_3'))
    model.add(Dropout(0.1, name='drop_3'))

    model.add(Conv2D(16, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                          kernel_initializer='he_normal', name='conv_4'))
    model.add(MaxPooling2D(pool_size=(3, 2), padding='same', name='mp_4'))

    model.add(Flatten())
    model.add(Dropout(0.4, name='drop_final'))
    model.add(Dense(300, activation='relu', name='dense_1'))
    model.add(Dense(num_classes, activation='softmax', name='dense_2'))

    opt = Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    if show_summary:
        model.summary()

    return model

def create_standard_CNN_allin1(name='standard', lr=0.0001, show_summary=False):
    # 16kHz
    # FIXME: changed to 2 seconds
    width_time = 1926  # 16kHz
    num_freq_bins = 128
    num_classes = 4
    model = Sequential()
    filters = (3, 3)

    model.add(Conv2D(32, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                          kernel_initializer='he_normal', input_shape=(width_time, num_freq_bins, 1), name='conv_1'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', name='mp_1'))
    model.add(BatchNormalization(name='batch_norm_1'))
    model.add(Dropout(0.1, name='drop_1'))

    model.add(Conv2D(32, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                          kernel_initializer='he_normal', name='conv_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', name='mp_2'))
    model.add(BatchNormalization(name='batch_norm_2'))
    model.add(Dropout(0.1, name='drop_2'))

    model.add(Conv2D(32, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                          kernel_initializer='he_normal', name='conv_3'))
    model.add(MaxPooling2D(pool_size=(3, 2), padding='same', name='mp_3'))
    model.add(BatchNormalization(name='batch_norm_3'))
    model.add(Dropout(0.1, name='drop_3'))

    model.add(Conv2D(32, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                          kernel_initializer='he_normal', name='conv_4'))
    model.add(MaxPooling2D(pool_size=(3, 2), padding='same', name='mp_4'))

    model.add(Flatten())
    model.add(Dropout(0.4, name='drop_final'))
    model.add(Dense(300, activation='relu', name='dense_1'))
    model.add(Dense(num_classes, activation='softmax', name='dense_2'))

    opt = Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    if show_summary:
        model.summary()

    return model

def create_standard_CNN_allin1V2(name='standard', lr=0.0001, show_summary=False):
    width_time = 48  # 16kHz
    num_freq_bins = 128
    num_classes = 4

    model = Sequential()
    filters = (3, 3)

    #model.add(Input(shape=(width_time, num_freq_bins, 1)))  # Ensure input shape is defined
    #model.add(CustomMasking(mask_value=np.inf))  # Add CustomMasking layer
    
    model.add(Conv2D(16, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                          kernel_initializer='he_normal', name='conv_1'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', name='mp_1'))
    model.add(BatchNormalization(name='batch_norm_1'))
    model.add(Dropout(0.1, name='drop_1'))

    model.add(Conv2D(16, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                          kernel_initializer='he_normal', name='conv_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', name='mp_2'))
    model.add(BatchNormalization(name='batch_norm_2'))
    model.add(Dropout(0.1, name='drop_2'))

    model.add(Conv2D(16, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                          kernel_initializer='he_normal', name='conv_3'))
    model.add(MaxPooling2D(pool_size=(3, 2), padding='same', name='mp_3'))
    model.add(BatchNormalization(name='batch_norm_3'))
    model.add(Dropout(0.1, name='drop_3'))

    model.add(Conv2D(16, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                          kernel_initializer='he_normal', name='conv_4'))
    model.add(MaxPooling2D(pool_size=(3, 2), padding='same', name='mp_4'))

    model.add(Flatten())
    model.add(Dropout(0.4, name='drop_final'))
    model.add(Dense(300, activation='relu', name='dense_1'))
    model.add(Dense(num_classes, activation='softmax', name='dense_2'))

    opt = Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    if show_summary:
        model.summary()
    return model

def create_CNN_mevd(lr=0.001, bidirectional=False):
    width_time = 48 
    model = Sequential()
    filters = (3, 3)
    model.add(TimeDistributed(Conv2D(16, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                                     kernel_initializer='he_normal'), input_shape=(None, width_time, 128, 1),
                              name='conv_1'))
    model.add(
        TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same', name='mp_1')))  # default poolsize = (2, 2), stride = (2,2)
    model.add(TimeDistributed(BatchNormalization(name='batch_norm_1')))
    model.add(Dropout(0.1, name='drop_1'))
 
    model.add(TimeDistributed(Conv2D(16, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                                     kernel_initializer='he_normal'), name='conv_2'))
    model.add(
        TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same', name='mp_2')))  # default poolsize = (2, 2), stride = (2,2)
    model.add(TimeDistributed(BatchNormalization(name='batch_norm_2')))
    model.add(Dropout(0.1, name='drop_2'))
 
    model.add(TimeDistributed((Conv2D(16, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                                      kernel_initializer='he_normal', name='conv_3'))))
    model.add(
        TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same', name='mp_3')))  # default poolsize = (2, 2), stride = (2,2)
    model.add(TimeDistributed(BatchNormalization(name='batch_norm_3')))
 
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dropout(0.4, name='drop_final')))
 
    if not bidirectional:
        model.add(LSTM(50, return_sequences=True, name='uni_lstm'))
    else:
        model.add(Bidirectional(LSTM(50, return_sequences=True, name='bi_lstm')))
        
    model.add(TimeDistributed(Dense(50, activation='relu', name='dense_1')))
    model.add(TimeDistributed(Dense(4, activation='softmax', name='dense_2')))
 
    opt = SGD(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
 
    print(model.summary())
    return model

# for the test 
class BaselineHyperModelCategoricalTVT(keras_tuner.HyperModel):
    def __init__(self, dir_path, x_train, y_train, x_validate, y_validate, objective_name, name=None, tunable=True):
        self.name = name
        self.tunable = tunable

        self.curr_trial = 0
        self.dir_path = dir_path
        self.x_train, self.y_train = x_train, y_train
        self.x_validate, self.y_validate = x_validate, y_validate
        self.objective_name = objective_name

        self._build = self.build
        self.build = self._build_wrapper

    def build(self, hp):
        width_time = 48 
        model = Sequential()
        filters = (3, 3)

        model.add(Input(shape=(None, width_time, 128, 1)))  # Ensure input shape is defined
        model.add(CustomMasking(mask_value=np.inf))  # Add CustomMasking layer

        model.add(TimeDistributed(Conv2D(16, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                                        kernel_initializer='he_normal'), name='conv_1'))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same', name='mp_1')))
        model.add(TimeDistributed(BatchNormalization(name='batch_norm_1')))
        model.add(Dropout(0.1, name='drop_1'))

        model.add(TimeDistributed(Conv2D(16, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                                        kernel_initializer='he_normal'), name='conv_2'))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same', name='mp_2')))
        model.add(TimeDistributed(BatchNormalization(name='batch_norm_2')))
        model.add(Dropout(0.1, name='drop_2'))

        model.add(TimeDistributed(Conv2D(16, filters, strides=(1, 1), activation='relu', data_format='channels_last',
                                        kernel_initializer='he_normal', name='conv_3')))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same', name='mp_3')))
        model.add(TimeDistributed(BatchNormalization(name='batch_norm_3')))

        model.add(TimeDistributed(Flatten()))
        model.add(TimeDistributed(Dropout(0.4, name='drop_final')))

        bidirectional = hp.Boolean("bidirectional", default=False)

        if not bidirectional:
            model.add(LSTM(50, return_sequences=True, name='uni_lstm'))
        else:
            model.add(Bidirectional(LSTM(50, return_sequences=True, name='bi_lstm')))

        model.add(TimeDistributed(Dense(50, activation='relu', name='dense_1')))
        model.add(TimeDistributed(Dense(4, activation='softmax', name='dense_2')))

        opt_name = hp.Choice("optimizer", ["sgd", "adam"])
        lr = hp.Float("learning_rate", min_value=1e-6, max_value=1e-1, step=10, sampling="log")

        if opt_name == "sgd":
            opt = SGD(lr)
        elif opt_name == "adam":
            opt = Adam(lr)
        else:
            print("Invalid optimizer choice")
            exit(-1)

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model

    def fit(self, hp, model, *args, **kwargs):
        # This function creates the necessary directories to save the optimization results and fits the model
        try:
            makedirs(self.dir_path + 'history/history_for_trial_' + str(self.curr_trial) + '/')
            makedirs(self.dir_path + 'times/training_times_for_trial_' + str(self.curr_trial) + '/')
            makedirs(self.dir_path + 'models/models_for_trial_' + str(self.curr_trial) + '/')
        except:
            pass
        
        start_train = time.time()
        # Batch size values were inferred from previous experiments on 4QAED
        # Another difference from fitting the model normally is the batch size search space definition, akin to what was
        # done in the build() function.
        tvt_hist = model.fit(x=self.x_train, y=self.y_train, 
                                validation_data=(self.x_validate, self.y_validate), 
                                *args, batch_size=hp.Int("batch_size", min_value=32, max_value=256, step=2, sampling='log'), **kwargs)
        end_train = time.time()

        # Save history, time to train and model for current tvt.
        with open(self.dir_path + 'history/history_for_trial_' + str(self.curr_trial) + '/' + str(EVAL_STRATEGY_PATH[:-1]) + '_hist.pickle', 'wb') as hist_tvt_file:
            pickle.dump(History(tvt_hist.history['accuracy'], tvt_hist.history['val_accuracy'], tvt_hist.history['loss'], tvt_hist.history['val_loss']), hist_tvt_file)

        with open(self.dir_path + 'times/training_times_for_trial_' + str(self.curr_trial) + '/' + str(EVAL_STRATEGY_PATH[:-1]) + '_training.txt', 'w') as train_time_file:
            train_time_file.write(str((end_train - start_train)/60))

        # Very important, as these need to be loaded for testing the "optimal" hyperparemeters found.
        model.save_weights(self.dir_path + 'models/models_for_trial_' + str(self.curr_trial) + '/' + 'model_' + str(EVAL_STRATEGY_PATH[:-1]))
        
        with open(self.dir_path + 'splits/' + str(EVAL_STRATEGY_PATH[:-1]) + '_results_for_trial_' + str(self.curr_trial) + '.txt', 'w') as results_file:
            results_file.write(str(tvt_hist.history[self.objective_name][-1])) 
        
        # Prepare for next trial.
        self.curr_trial = self.curr_trial + 1

        # The objective function result is returned to adjust parameters for the next optimization trial.
        return {self.objective_name: tvt_hist.history[self.objective_name][-1]}



def my_hypertuner_bcnn_tvt(X, y, tuner_type="bayesian"): 
    # Create necessary directories to save optimization results
    try:
        makedirs(EVAL_STRATEGY_PATH + 'splits/')
    except:
        pass

    try:
        makedirs(EVAL_STRATEGY_PATH + 'history')
    except:
        pass

    try:
        makedirs(EVAL_STRATEGY_PATH + 'models/')
    except:
        pass

    try:
        makedirs(EVAL_STRATEGY_PATH + 'times')
    except:
        pass

    # Define objective function. We simply choose "training accuracy", or just "accuracy", and
    # ask for the optimizer to maximize this function.
    tuner_objectives = ['accuracy']
    hypemodel_objective = keras_tuner.Objective(tuner_objectives[0], direction="max")
    x_test=X
    y_test=y
    ### Split training data preparation for current fold ###
    WINDOW_WIDTH = 1926
    x_test_chunk, y_test_chunk = [], []
    #print(x_train[:100])
    # Iterate over each array in x_train and y_train
    for i,(x_array, y_array) in enumerate(zip(x_test, y_test)):
        for arr in x_array:
            # Flatten x_array and add its elements to x_train_chunk
            #print(arr.shape)
            x_test_chunk.append(arr)
            y_test_chunk.append(y_array)


    # Convert lists to numpy arrays and reshape x_train_chunk
    x_test_chunk = np.asarray(x_test_chunk).reshape(len(x_test_chunk), WINDOW_WIDTH, 128, 1)
    y_test_chunk = np.asarray(x_test_chunk).reshape(len(x_test_chunk), 4)  

    
    # Create custom HyperModel class. We pass the relevant parameters for conducting optimization as well.
    hypemodel = BaselineHyperModelCategoricalTVT(dir_path=EVAL_STRATEGY_PATH,
        x_test = x_test_chunk, y_test=y_test_chunk,
        objective_name='accuracy')

    # Create directory to save the optimization results dictionary. Will be used for
    # finding the best ranked trials, or in other words, sort trials from best to worst
    # according to the objective function.
    summary_dir = tuner_type + '_tuner_summary/'
    try:
        makedirs(EVAL_STRATEGY_PATH + summary_dir)
    except FileExistsError:
        pass

    # Set up EarlyStopping.
    # The model will either stop training if the accuracy treshold is reached, preventing overfitting,
    # or if the model is not improving at all for a significant amount of time, to prevent wasting resources.
    th = MyCallbacks(threshold=TRAIN_ACCURACY_TRESHOLD)
    es = EarlyStopping(monitor='val_accuracy', patience=NO_IMPROVEMENT_OVER_EPOCHS)
    # Set tuner as None to ensure one of either hyperband or bayesian is used
    # For testing purposes, considere only bayesian for now.
    tuner = None

    print('Preparing tuner...')

    # TODO: Keras Tuner's implementation of Hyperband is buggy, need to implement my own
    # FIXME: Change max_trials and epochs after testing
    if tuner_type=="hyperband":
        print('Using Hyperband Optimizer...')
        tuner = keras_tuner.Hyperband(
            hypemodel,
            objective=hypemodel_objective,
            factor=3, hyperband_iterations=2, seed=1, overwrite=True,
            directory=EVAL_STRATEGY_PATH + "tuner_logs", project_name="hyperband_opt_bi_modal_complete"
        )
    elif tuner_type=="bayesian":
        print('Using Bayesian Optimizer...')
        tuner = keras_tuner.BayesianOptimization(
            hypemodel,
            objective=hypemodel_objective,
            max_trials=10, seed=1, overwrite=True,
            directory=EVAL_STRATEGY_PATH + "tuner_logs", project_name="bayesian_opt_bi_modal_complete" 
        )

    # Define search parameters.
    tuner.search(epochs=200, callbacks=[th, es], initial_epoch=0)

    # Save all trials optimization results.
    tuner_summary = tuner.oracle.get_best_trials(num_trials=10)
    
    with open(EVAL_STRATEGY_PATH + summary_dir + tuner_type + '_cv_summary.pickle', 'wb') as f:
        results_list = []
        for rank, trial in enumerate(tuner_summary):
                trial_sum = {'Rank': rank,
                             'ID': trial.trial_id, 'Hyperparameters': trial.hyperparameters.values,
                             'Objective': tuner_objectives, 'Score': trial.score}
                results_list.append(trial_sum)
        pickle.dump(results_list, f)
    
    print('Done tuning model with cross-validation strategy!')

def test_model_multi_tvt_eval_all(X_test, labels_test):
    print("Start model testing:")

    # Start by loading the optimization dictionary saved earlier.
    with open(EVAL_STRATEGY_PATH + PATH_TO_TRIALS_SUMMARY + '.pickle', 'rb') as file_with_trial_sum:
        trial_dict = pickle.load(file_with_trial_sum)

    # For each trial in the dictionary, directories to save the relevant metrics are created, the corresponding hyperparameters
    # are loaded alongside the model saved for that trial. The model is evaluated, with no modifications from the earlier
    # optimization trial, using the test set of the same TVT split used for optimization. Results are saved to the created directories.
    for curr_trial_test in range(len(trial_dict)):
        curr_trial_test_path = EVAL_STRATEGY_PATH + 'results_trial_rank_' + str(curr_trial_test) + '/'

        try:
            makedirs(curr_trial_test_path)
        except FileExistsError:
            pass

        #TODO: Same as in run_f1_score_single
        loss, macro_f1_score, scores_all = [], [], []
        pscores, rscores, pscores_each, rscores_each = [], [], [], []

        f1_each = []
        confusion_matrix_global = np.zeros((4, 4))
        # save the model with the highest f1_score

        # Get current trial
        curr_trial_dict = trial_dict[curr_trial_test]
        this_id = int(curr_trial_dict['ID'])
        these_params = curr_trial_dict['Hyperparameters']

        create_dirs_for_model_test([these_params['batch_size'], these_params['optimizer'], these_params['learning_rate'], 'results_trial_rank_' + str(curr_trial_test) + '/'], cv_eval=False, eval_all_trials=True)

        try:
            makedirs(curr_trial_test_path + 'times/testing_times/')
        except FileExistsError:
            pass

        # Build model and load weights for current fold
        model = create_standard_CNN()
        model.load_weights(EVAL_STRATEGY_PATH + 'models/models_for_trial_' + str(this_id) + '/model_' + str(EVAL_STRATEGY_PATH[:-1]))

        ## Testing portion ##
        print('Loaded model for ' + str(EVAL_STRATEGY_PATH[:-1]) + '. Testing...')
        WINDOW_WIDTH = 1926
        x_test_chunk, y_test_chunk = [], []

        for x_array, y_array in zip(X_test, labels_test):
            for arr in x_array:
                # Flatten x_array and add its elements to x_train_chunk
                x_test_chunk.append(arr)
                y_test_chunk.append(y_array)
        
        x_test_chunk = np.asarray(x_test_chunk).reshape(len(x_test_chunk), WINDOW_WIDTH, 128, 1)
        #x_test_chunk = np.expand_dims(x_test_chunk, axis=1)
        y_test_chunk = np.asarray(y_test_chunk).reshape(len(y_test_chunk), 4)  
        #y_test_chunk = np.expand_dims(y_test_chunk, axis=1)

        # Evaluate the model
        start_inference = time.time()
        score_temp = model.evaluate(
            x_test_chunk, y_test_chunk, verbose=0)
        end_inference = time.time()
        with open(curr_trial_test_path + 'times/testing_times/' + EVAL_STRATEGY_PATH[:-1] + '_inference_time.txt', 'w') as testing_time:
            testing_time.write(str((end_inference - start_inference)/ 60))

        print(score_temp)
        scores_all.append(score_temp)

        # Predict test fold
        ypred_all = model.predict_on_batch(x=x_test_chunk)
        #ypred_all = np.squeeze(ypred_all)
        #y_test_chunk = np.squeeze(y_test_chunk)
        quad_real = np.asarray(y_test_chunk).argmax(axis=1)
        quad_pred = np.asarray(ypred_all).argmax(axis=1)

        # Calculate Confusion Matrix
        print(confusion_matrix(quad_real, quad_pred))
        conf_temp = confusion_matrix(quad_real, quad_pred)
        confusion_matrix_global = confusion_matrix_global + conf_temp

        # Predict for F1 Score, Precision, Recall
        macro_f1_score.append(f1_score(quad_real, quad_pred, average='weighted'))
        pscores.append(precision_score(quad_real, quad_pred, average='weighted'))
        rscores.append(recall_score(quad_real, quad_pred, average='weighted'))

        # Predict each class F1 Score, Precision, Recall
        f1_each.append(f1_score(quad_real, quad_pred, average=None))
        pscores_each.append(precision_score(quad_real, quad_pred, average=None))
        rscores_each.append(recall_score(quad_real, quad_pred, average=None))

        # True and pred saved as [[num_labels][arousal_true, valence_true],[num_labels][arousal_pred, valence_pred]]
        with open(curr_trial_test_path + 'true_and_pred/' + str(EVAL_STRATEGY_PATH[:-1]) + '.pickle', 'wb') as file_true_pred:
            pickle.dump([quad_real, quad_pred], file_true_pred) 

        f1s_name = 'f1_score_multi_modal_categorical_bs_' + str(these_params['batch_size']) + '_opt_' + str(these_params['optimizer']) + '_lr_' + str(these_params['learning_rate'])
        precs_name = 'precision_multi_modal_categorical_bs_' + str(these_params['batch_size']) + '_opt_' + str(these_params['optimizer']) + '_lr_' + str(these_params['learning_rate'])
        recs_name = 'recall_multi_modal_categorical_bs_' + str(these_params['batch_size']) + '_opt_' + str(these_params['optimizer']) + '_lr_' + str(these_params['learning_rate'])

        f1e_name = 'f1_each_multi_modal_categorical_bs_' + str(these_params['batch_size']) + '_opt_' + str(these_params['optimizer']) + '_lr_' + str(these_params['learning_rate'])
        precse_name = 'precision_each_multi_modal_categorical_bs_' + str(these_params['batch_size']) + '_opt_' + str(these_params['optimizer']) + '_lr_' + str(these_params['learning_rate'])
        recse_name = 'recall_each_multi_modal_categorical_bs_' + str(these_params['batch_size']) + '_opt_' + str(these_params['optimizer']) + '_lr_' + str(these_params['learning_rate'])

        conf_name = 'conf_mat_multi_modal_categorical_bs_' + str(these_params['batch_size']) + '_opt_' + str(these_params['optimizer']) + '_lr_' + str(these_params['learning_rate'])
        scores_name = 'scores_multi_modal_categorical_bs_' + str(these_params['batch_size']) + '_opt_' + str(these_params['optimizer']) + '_lr_' + str(these_params['learning_rate'])

        with open(curr_trial_test_path + 'f1_macro_best100/' + f1s_name + '.pickle', 'wb') as file_with_f1:
            pickle.dump(macro_f1_score[0], file_with_f1)
        with open(curr_trial_test_path + 'precision/' + precs_name + '.pickle', 'wb') as file_with_precision:
            pickle.dump(pscores[0], file_with_precision)
        with open(curr_trial_test_path + 'recall/' + recs_name + '.pickle', 'wb') as file_with_recall:
            pickle.dump(rscores[0], file_with_recall)

        with open(curr_trial_test_path + 'f1_each/' + f1e_name + '.pickle', 'wb') as file_with_each:
            pickle.dump(f1_each[0], file_with_each)
        with open(curr_trial_test_path + 'precision_each/' + precse_name + '.pickle', 'wb') as file_with_precision_each:
            pickle.dump(pscores_each[0], file_with_precision_each)
        with open(curr_trial_test_path + 'recall_each/' + recse_name + '.pickle', 'wb') as file_with_recall_each:
            pickle.dump(rscores_each[0], file_with_recall_each)

        with open(curr_trial_test_path + 'conf_matrix/' + conf_name + '.pickle', 'wb') as file_with_conf:
            pickle.dump(confusion_matrix_global, file_with_conf)
        with open(curr_trial_test_path + 'scores/' + scores_name + '.pickle', 'wb') as file_with_scores:
            pickle.dump(scores_all, file_with_scores)

        with open(curr_trial_test_path + 'trial_' + str(this_id) + '_hyperparameters.txt', '+a') as hp_file:
            hp_file.write(str(these_params))

def avg(lst):
    return sum(lst) / len(lst)

def find_index_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def calculate_column_medians(array):
    # Ensure the input is a NumPy array
    array = np.array(array)
    
    # Calculate the median along the columns (axis 0)
    column_medians = np.median(array, axis=0)
    
    # Return the column medians as a 1D array
    return column_medians

def write_to_file(list, file_write, missing_quadrants, mean=False):
    if mean:
        file_write.write("Mean: [")
    else:
        file_write.write("Std: [")
    indice = 0
    for i in range(4):
        quadrant_value = "Q%d" % (i+1)


        if quadrant_value in missing_quadrants:
            value = 0
        else:
            if mean:
                value = np.mean(list[indice]) * 100
            else:
                value = np.std(list[indice]) * 100
            indice +=1
        

        file_write.write(f"{value:.2f}%")
        if i < 3:
            file_write.write("; ")
    file_write.write("]\n")

def check_and_clean_data(data, data_name):
    # Check for NaN values
    for i, sample in enumerate(data):
        if np.isnan(sample).any():
            print(f"NaN values found in {data_name} at sample index: {i}")
            raise ValueError(f"NaN values found in {data_name}")

    # Normalize the data if not already normalized
    data = np.array(data, dtype=np.float32)
    data = (data - np.mean(data)) / np.std(data)
    return data

def evaluate_and_print(X, y, set_name,flag, learning_rate, file_write, model=None):

        if flag == 0:
            if X.ndim == 3:
                X = np.expand_dims(X, axis=0)
            y_pred = model.predict_on_batch(x=X)
            return y_pred
            #conf_mat = confusion_matrix(y, y_pred)
        elif flag ==1:
            y_pred = y
            y = X
            #in this case the song_set is a index so i can retieve the right name of the file 
            song_name = get_file_name_by_index(set_name)
            #the flag 0 indicates that the conf will be done per song
            conf_mat = process_confusion_matrix(COMPARISON_SEG_OUTPUT,song_name,0)
            folder_path = COMPARISON_SEG_OUTPUT
            file_path = os.path.join(folder_path, song_name)
            y_pred = filter_outliers(file_path, y_pred, threshold=4.0)
            #print(set_name)
            #print("OLA")
            #print(conf_mat)

        else:
            y_pred = y
            y = X
            #the flag 1 indicates that the conf will be done for all songs
            gather_all_data(COMPARISON_SEG_OUTPUT, COMPARISON_SEG_OUTPUT_ALL)
            conf_mat = process_confusion_matrix(COMPARISON_SEG_OUTPUT_ALL,set_name,1)
            y_pred = filter_outliers(COMPARISON_SEG_OUTPUT_ALL, y_pred, threshold=4.0)
        #y_pred = remove_outliers(y_pred)
        #conf_mat = confusion_matrix(y, y_pred)
      
        row_sums = conf_mat.sum(axis=1, keepdims=True)  # Calculate row sums
        conf_mat_percentage = conf_mat / row_sums * 100  # Convert to percentages
        f1_scores = f1_score(y, y_pred, average=None)  # F1-score for each class
        f1_global = f1_score(y, y_pred, average='weighted')  # Overall F1-score
        recall_scores = recall_score(y, y_pred,average=None)
        recall_global = recall_score(y, y_pred,average='weighted')
        precision_scores = precision_score(y,y_pred,average=None)
        precision_global = precision_score(y,y_pred,average='weighted')
        accuracy = accuracy_score(y, y_pred)

        quadrants = list(np.unique(y))
        all_quadrants = ['Q1','Q2', 'Q3', 'Q4']
        missing_quadrants = list(set(all_quadrants) - set(quadrants))
       
        # Printing and Writing Results
        if file_write is not None and set_name != "Test":
            file_write.write("\n\n------------------------------\n\n")
            file_write.write("F1-Score\n\n")
            file_write.write("Mean: %.2f%%\n" % (f1_global*100))

            file_write.write("\nPrecision\n\n")
            file_write.write("Mean: %.2f%%\n" % (precision_global*100))

            file_write.write("\nRecall\n\n")
            file_write.write("Mean: %.2f%%\n" % (recall_global*100))

            # now i want to file_write.write f1-score like:
            #"Mean: [//%; //%; //%; //%]
            #Std: [//%; //%; //%; //%]"

            file_write.write("\nF1-Score per quadrant\n\n")
            print(f1_scores)
            write_to_file(f1_scores, file_write, missing_quadrants, mean=True)

            file_write.write("\nPrecision per quadrant\n\n")
            write_to_file(precision_scores, file_write, missing_quadrants, mean=True)

            file_write.write("\nRecall2 per quadrant\n\n")
            
            write_to_file(recall_scores, file_write, missing_quadrants, mean=True)

            # now i wanna file_write.write the confusion matrix, just mean and then just std
            '''
            [//%; //%; //%; //%]
            [//%; //%; //%; //%]
            [//%; //%; //%; //%]
            [//%; //%; //%; //%]
            '''

            # Convert confusion matrix to percentages
            conf_mat_percentage = conf_mat / conf_mat.sum(axis=1, keepdims=True) * 100

            # Fill missing entries to make the confusion matrix 4x4
            if conf_mat_percentage.shape[0] < 4:
                conf_mat_percentage = np.pad(conf_mat_percentage, ((0, 4 - conf_mat_percentage.shape[0]), (0, 0)), mode='constant', constant_values=0)
            if conf_mat_percentage.shape[1] < 4:
                conf_mat_percentage = np.pad(conf_mat_percentage, ((0, 0), (0, 4 - conf_mat_percentage.shape[1])), mode='constant', constant_values=0)

            # Writing confusion matrix percentages to a file
            file_write.write("\nConfusion Matrix (in %)\n\n")
            for row in conf_mat_percentage:
                file_write.write("[")
                for value in row:
                    if np.isnan(value):
                        file_write.write("0.00%")
                    else:
                        file_write.write(f"{value:.2f}%")
                    file_write.write("; ")
                file_write.write("]\n")

        print(f"Metrics for {set_name} set:")
        print(f"Overall F1-score: {f1_global:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print("F1-scores per class:")
        for i, f1 in enumerate(f1_scores, 1):
            print(f" Q{i}: {f1:.3f}")
        print(conf_mat)

        if set_name == "Test":
            file_write.write("\n\n------------------------------\n\n")
            file_write.write("F1-Score\n\n")
            file_write.write("Mean: %.2f%%\n" % (f1_global*100))

            file_write.write("\nPrecision\n\n")
            file_write.write("Mean: %.2f%%\n" % (precision_global*100))

            file_write.write("\nRecall\n\n")
            file_write.write("Mean: %.2f%%\n" % (recall_global*100))

            # now i want to file_write.write f1-score like:
            #"Mean: [//%; //%; //%; //%]
            #Std: [//%; //%; //%; //%]"

            file_write.write("\nF1-Score per quadrant\n\n")
            write_to_file(f1_scores,file_write,[] , mean=True)

            file_write.write("\nPrecision per quadrant\n\n")
            write_to_file(precision_scores, file_write, [] , mean=True)

            file_write.write("\nRecall per quadrant\n\n")
            
            write_to_file(recall_scores, file_write,[] , mean=True)

            # now i wanna file_write.write the confusion matrix, just mean and then just std
            '''
            [//%; //%; //%; //%]
            [//%; //%; //%; //%]
            [//%; //%; //%; //%]
            [//%; //%; //%; //%]
            '''

            file_write.write("\nConfusion Matrix (in %)\n\n")
            for i in range(4):
                file_write.write("[")
                for j in range(4):
                    # Format each matrix element as a percentage with two decimal places
                    file_write.write(f"{conf_mat_percentage[i, j]:.2f}%")
                    if j < 3:
                        file_write.write("; ")
                file_write.write("]\n")  # This adds a semicolon and a space after each row except the last one

            file_write.write("\n\n------------------------------\n\n")
            file_write.flush()
        return y_pred

def run_f1score_tvt_split(X_train, labels_train, X_test, labels_test, epochs, batch_size, learning_rate,song_names_array):
    WINDOW_WIDTH = 48
    print("começo a treinar:")
    # F1 Score with Train-Validation-Test Split
    # either 70-15-15 or 40-30-30 depending on
    # EVAL_STRATEGY_PATH variable
    acum, loss, macro_f1_score, scores_all = 0, [], [], []
    pscores, rscores, pscores_each, rscores_each = [], [], [], []
 
    history_all_accuracy = []
    history_all_accuracy_val = []
    history_all_loss = []
    history_all_loss_val = []
    f1_each = []
    confusion_matrix_global = np.zeros((4, 4))
    # save the model with the highest f1_score
    max_f1 = 0
    ###############################################
    ## Callbacks ##################################
    ###############################################
    th = MyCallbacks(threshold=0.9)
    ###############################################
 
 
    ###############################################
    ## Training portion ##

    print('Training...')
    print('Clearing prev model.')

    """x_test_chunk = []
    y_test_chunk = []

    for i,(x_array, y_array) in enumerate(zip(X_test, labels_test)):
        for arr_x in x_array:
            x_test_chunk.append(arr_x)
        for arr_y in y_array:
            y_test_chunk.append(arr_y)
    
    reshaped_chunks = []
    for segment in x_test_chunk:
        reshaped_chunks.append(segment)

    # Combine reshaped chunks into the final array, if needed
    x_test_chunk = np.asarray(reshaped_chunks)

    for arr_x,arr_y in zip(x_test_chunk,y_test_chunk):
        arr_x = np.asarray(arr_x)
        arr_x = arr_x.reshape(-1, 48, 128, 1)
        print(f"Original arr_x shape: {arr_x.shape}")
        if arr_x.ndim == 3:
            arr_x = np.expand_dims(arr_x, axis=0)  # Add batch dimension if missing
        print(f"Reshaped arr_x shape: {arr_x.shape}")
    exit(-1)"""

    # re-initilize and compile
    # the arr cotains the info of a segment cotaining mini segments in a specific song
    """test1 = []
    test2 = []
    for i,(x_array, y_array) in enumerate(zip(X_test, labels_test)):
            # Flatten x_array and add its elements to x_train_chunk
            #print(arr.shape)
        for arr in x_array:
            test1.append(arr)
        for arr_y in y_array:
            test2.append(arr_y)
    print(len(test1))
    exit(-1)"""

    x_train_chunk, y_train_chunk = [], []

    for i,(x_array, y_array) in enumerate(zip(X_train, labels_train)):
        for arr in x_array:
            # Flatten x_array and add its elements to x_train_chunk
            #print(arr.shape)
            x_train_chunk.append(arr)
            y_train_chunk.append(y_array)


    # Convert lists to numpy arrays and reshape x_train_chunk
    x_train_chunk = np.asarray(x_train_chunk).reshape(len(x_train_chunk), WINDOW_WIDTH, 128, 1)
    #x_train_chunk = np.expand_dims(x_train_chunk, axis=1)
    y_train_chunk = np.asarray(y_train_chunk).reshape(len(y_train_chunk), 4)
    #y_train_chunk = np.expand_dims(y_train_chunk, axis=1)  
    model = create_standard_CNN_allin1V2(lr=learning_rate)
    start_training = time.time()
    # train and save to log file
    history = model.fit(x=x_train_chunk, y=y_train_chunk,
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[th], verbose=1)
    end_training = time.time()
    with open(EVAL_STRATEGY_PATH + 'times/training_time.txt', 'w') as training_time:
        training_time.write(str((end_training - start_training)/ 60))
 
    print("\nLoss on last epoch:")
    print(history.history['loss'][-1])
    print("\nAcc on last epoch:")
    print(history.history['accuracy'][-1])
    loss.append(history.history['loss'][-1])
 
    # Save accuracy and loss from the training phase
    aux = np.array([-1 for _ in range(len(history.history['accuracy']))])
    history_all_accuracy.append(history.history['accuracy'])
    history_all_accuracy_val.append(aux)
    history_all_loss.append(history.history['loss'])
    history_all_loss_val.append(aux)
    
    ###############################################
    ## Testing portion ##
    print('Testing...')
    create_dirs_for_model_test([epochs, batch_size, learning_rate], cv_eval=False, eval_all_trials=False)
    x_test_chunk, y_test_chunk = [], []
    x_test_full, y_test_full = [], []
    #x_test_chunk = [ar for x_array in X_test for arr in x_array for ar in arr] 
    #y_test_chunk = [arr_y for y_array in labels_test for arr_y in y_array]

    
    """for index,current_sample in enumerate(X_test):
        x_test_full.append(np.asarray(current_sample).reshape(len(current_sample), WINDOW_WIDTH, 128, 1))
        y_test_full.append(labels_test[index])


    x_test_full, y_test_full = np.asarray(x_test_full), np.asarray(y_test_full)"""
    #x_test_full = np.expand_dims(x_test_full, axis=1)
    #y_test_full = np.expand_dims(y_test_full, axis=1)


    for i,(x_array, y_array) in enumerate(zip(X_test, labels_test)):
        for arr_x in x_array:
            x_test_chunk.append(arr_x)
        for arr_y in y_array:
            y_test_chunk.append(arr_y)
    
    reshaped_chunks = []
    for segment in x_test_chunk:
        reshaped_chunks.append(segment)

    # Combine reshaped chunks into the final array, if needed
    x_test_chunk = np.asarray(reshaped_chunks)        
    #x_test_chunk = np.expand_dims(x_test_chunk, axis=1)
    y_test_chunk = np.asarray(y_test_chunk).reshape(len(y_test_chunk), 4)  
    #y_test_chunk = np.expand_dims(y_test_chunk, axis=1)
    
    # Evaluate the model
    #start_time = time.time()
    #score_temp = model.evaluate(x=x_test_chunk, y=y_test_chunk, verbose=0)
    #end_inference = time.time()

    #print(score_temp)
    #scores_all.append(score_temp)
    file_write = open("test_results.txt","w")

    #X_test = np.array([X_test])
    y_pred_test =[]
    print("\n--------------------------------------\n")
    for arr_x,arr_y in zip(x_test_chunk,y_test_chunk):
        #print("OLLOALAOLAAOALALALAALL")
        #print(arr_x)
        #print(arr_y)
        arr_x = np.asarray(arr_x)
        arr_x = arr_x.reshape(-1, 48, 128, 1)
        result_per_segment = evaluate_and_print(arr_x, arr_y, "Pred", 0, learning_rate,file_write=file_write, model=model)
        #result = calculate_column_medians(result_per_segment)
        #y_pred_test.append(result)
        print(result_per_segment)
        print("mudou")
    #print("acabou")
    #print(y_pred_test)
    exit(-1)
    
    #y_pred_song = evaluate_and_print(x_test_chunk, y_test_chunk, "Pred", 0, learning_rate,file_write=file_write, model=model)
    print("\n--------------------------------------\n")
    end_time = time.time()
    y_pred_song = y_pred_test
    # Segmentation correction
    print(y_pred_song)
    #y_pred_song = get_annotation(y_pred_test,0)
    print(len(y_pred_song))
    print("OLAAOLAO")
    for arr in y_pred_song:
        print((arr))
    
    # here you put the percentage of the y pred into a int number this is into 1,2,3 or 4
    y_pred_song_fixed = [['Q' + str(np.argmax(row) + 1) for row in arr] for arr in y_pred_song]
    write_pred_allin1(y_pred_song_fixed)
    process_files(REAL_FOLDER, PRED_FOLDER, COMPARISON_SEG_OUTPUT)
    y_real_pred = extract_quadrants(COMPARISON_SEG_OUTPUT)
    y_pred_song_with_corretion =  get_annotation(y_real_pred[1],1)
    y_real_song_with_corretion = get_annotation(y_real_pred[0],1)
    y_pred_test = evaluate_and_print(y_real_pred[0], y_real_pred[1] , "Test", 2, learning_rate,file_write=file_write)  
    
    #y_real_song = get_annotation(y_test)
    #x_song = get_annotation(X_test)
    #print(f"Total time: {(end_time - start_time) / 60} minutes")
    for i in range(len(song_names_array)):
        with open('results/pred_allin1_quadrants/pred_real_per_music_'+ song_names_array[i] + '.pickle', 'wb') as res:
            pickle.dump([y_pred_song_with_corretion[i], y_real_song_with_corretion[i]], res)

        #arrays_numpy = np.array(x_song[i])
        quadrants_numpy_pred = np.array(y_pred_song_with_corretion[i])
        quadrants_numpy_real = np.array(y_real_song_with_corretion[i])
        song_id = song_names_array[i]
        pred_file_path = os.path.join('results', 'pred_allin1_metrics', f'pred_real_per_music_{song_id}.txt')

        pred_file_path_pointer = open(pred_file_path,"w")
            
        print("Outputting to %s" % (pred_file_path))
        y_pred = evaluate_and_print(quadrants_numpy_real, quadrants_numpy_pred, i,1, learning_rate,file_write=pred_file_path_pointer)
        pred_file_path_pointer.close()
 
    # save results
    # get the average f1 score
    #idx = find_index_nearest(macro_f1_score, avg(macro_f1_score))
    hist = History(history_all_accuracy,
                           history_all_accuracy_val,
                           history_all_loss,
                           history_all_loss_val)
 
    return hist, macro_f1_score, f1_each, confusion_matrix_global, scores_all, pscores, rscores, pscores_each, rscores_each

def get_file_name_by_index(index):
    folder_path = "real_pred_seg_correction_allin1"

    # Get list of files inside "real_pred_seg_correction_allin1"
    files = sorted([file for file in os.listdir(folder_path) if file != ".DS_Store"])

    if index < 0 or index >= len(files):
        return "Invalid index"

    return files[index]

def get_annotation(Y_array, flag):
    if flag == 0:
        quadrant_folder = "/welmo-data/tomasferreira/CNN_Data/quadrant_annotation_allin1"
    else:
        quadrant_folder = "real_pred_seg_correction_allin1"
    file_list = sorted(os.listdir(quadrant_folder), key=str.lower)

    quadrants_all_song = []  # Initialize a 2D list to store quadrants for each song

    for filename in file_list:
        if filename.endswith(".txt"):
        # print("Reading file:", filename)
            annotation_file_path = os.path.join(quadrant_folder, filename)
            num_segments = sum(1 for line in open(annotation_file_path) if line.strip())

            # Extract quadrants for the current song based on the number of segments
            quadrants = Y_array[:num_segments]
            Y_array = Y_array[num_segments:]

            # Append the quadrants for the current song to the 2D array
            quadrants_all_song.append(quadrants)

    # Print the 2D array containing quadrants for each song
    return quadrants_all_song

def write_pred_allin1(y_pred_song):
    folder_path = "struct_allin1"
    output_folder = "pred_annot_allin1"
    
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print("The 'struct' folder does not exist.")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    contents = os.listdir(folder_path)
    contents = [file for file in contents if file != ".DS_Store"]  # Filter out .DS_Store
    contents.sort(key=lambda x: x.lower())  # Sort contents alphabetically
    
    for i, filename in enumerate(contents):
        if filename.endswith(".json"):
            song_name = filename[:-5]  # Remove .json extension
            file_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, f"{song_name}.txt")
            
            with open(file_path, 'r') as file:
                data = json.load(file)
            segments = data["segments"]
            
            with open(output_path, 'w') as output_file:
                for segment, prediction in zip(segments, y_pred_song[i]):
                    print(prediction)
                    start_time = segment["start"]
                    end_time = segment["end"]
                    output_file.write(f"{start_time:05.2f} {end_time:05.2f} {prediction}\n")

def get_last_end_time(pred_annot_file):
    # Read the last line of the file
    with open(pred_annot_file, 'r') as file:
        lines = file.readlines()
        if lines:
            last_line = lines[-1].strip()
            parts = last_line.split()
            if len(parts) >= 2:
                last_end_time = float(parts[1])
                return last_end_time
    return None

def process_files(real_folder, pred_folder, output_folder):
    real_contents = os.listdir(real_folder)
    pred_contents = os.listdir(pred_folder)

    # Filter out .DS_Store
    real_contents = [file for file in real_contents if file != ".DS_Store"]
    pred_contents = [file for file in pred_contents if file != ".DS_Store"]

    real_contents.sort(key=lambda x: x.lower())
    pred_contents.sort(key=lambda x: x.lower())

    for real_file, pred_file in zip(real_contents, pred_contents):
        real_path = os.path.join(real_folder, real_file)
        pred_path = os.path.join(pred_folder, pred_file)

        combined_lines = process_single_file(real_path, pred_path)

        # Write output to file
        output_file_path = os.path.join(output_folder, real_file)
        with open(output_file_path, 'w') as output_file:
            for line in combined_lines:
                output_file.write(line)
                

def process_single_file(real_path, pred_path):
    real_lines = []
    pred_lines = []

    # Read lines from real_annot_allin1
    with open(real_path, 'r') as real_file:
        real_lines = real_file.readlines()

    # Read lines from pred_annot_allin1
    with open(pred_path, 'r') as pred_file:
        pred_lines = pred_file.readlines()

    combined_lines = []
    real_index = 0
    pred_index = 0

    # Initialize previous end times for real and pred segments
    prev_real_end_time = float('inf')
    prev_pred_end_time = float('inf')

    if real_lines:
        real_parts = real_lines[0].strip().split()
        prev_real_end_time = float(real_parts[1])
        prev_real_quadrant = real_parts[2]

    if pred_lines:
        pred_parts = pred_lines[0].strip().split()
        prev_pred_end_time = float(pred_parts[1])

    # Get the path of the predicted annotations file for the current song
    last_pred_end_time = get_last_end_time(pred_path)

    last_line_written = None

    while real_index < len(real_lines) or pred_index < len(pred_lines):
        if real_index < len(real_lines):
            real_parts = real_lines[real_index].strip().split()
            real_start_time = float(real_parts[0])
            real_end_time = float(real_parts[1])
            real_quadrant = real_parts[2]  # Preserve quadrant from real annotation
        else:
            real_start_time = float('inf')
            real_end_time = float('inf')

        if pred_index < len(pred_lines):
            pred_parts = pred_lines[pred_index].strip().split()
            pred_start_time = float(pred_parts[0])
            pred_end_time = float(pred_parts[1])
            pred_quadrant = pred_parts[2][1:]
        else:
            pred_start_time = float('inf')
            pred_end_time = float('inf')

        # Process segments based on start time
        if real_start_time <= pred_start_time:
            if prev_pred_end_time < real_start_time:
                combined_lines.append(f"{real_start_time:.2f} {real_end_time:.2f} {real_quadrant},Q{pred_quadrant}\n")
            else:
                end_time = min(real_end_time, prev_pred_end_time, last_pred_end_time)
                if end_time <= last_pred_end_time:  # Ensure we don't exceed the limit
                    line = f"{real_start_time:.2f} {end_time:.2f} {real_quadrant},Q{pred_quadrant}\n"
                    if line.strip() != last_line_written:
                        combined_lines.append(line)
                        last_line_written = line.strip()
                if end_time == last_pred_end_time:
                    break  # Break if end time exceeds last_pred_end_time

            prev_real_end_time = real_end_time
            prev_real_quadrant = real_quadrant
            real_index += 1
        else:
            if prev_real_end_time < pred_start_time:
                combined_lines.append(f"{pred_start_time:.2f} {pred_end_time:.2f} {prev_real_quadrant},Q{pred_quadrant}\n")
            else:
                end_time = min(pred_end_time, prev_real_end_time, last_pred_end_time)
                if end_time <= last_pred_end_time:  # Ensure we don't exceed the limit
                    line = f"{pred_start_time:.2f} {end_time:.2f} {prev_real_quadrant},Q{pred_quadrant}\n"
                    if line.strip() != last_line_written:
                        combined_lines.append(line)
                        last_line_written = line.strip()
                if end_time == last_pred_end_time:
                    break  # Break if end time exceeds last_pred_end_time

            prev_pred_end_time = pred_end_time
            pred_index += 1

    return combined_lines


def extract_quadrants(folder_path):
    all_quadrant_real = []
    all_quadrant_pred = []
    
    # Get a list of all txt files in the folder sorted alphabetically (case-insensitive)
    file_list = sorted(os.listdir(folder_path), key=lambda x: x.lower())
    
    for filename in file_list:
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r') as file:
                for line in file:
                    # Split the line by space to get parts
                    parts = line.strip().split(" ")
                    quadrant_real = parts[2].split(',')[0]
                    quadrant_pred = parts[2].split(',')[1]
                    # Append "Q" to each number in quadrant_real
                    quadrant_real = "Q" + quadrant_real
                    # Append quadrantReal and quadrantPred to their respective lists
                    all_quadrant_real.append(quadrant_real)
                    all_quadrant_pred.append(quadrant_pred)
    
    # Convert lists to numpy arrays
    result_array = np.array(all_quadrant_real), np.array(all_quadrant_pred)
    
    return result_array

def process_confusion_matrix(folder_path, song_id, flag):
    """
    Reads the annotation data from a file, calculates the confusion matrix,
    and formats it as a string.
    
    Parameters:
    folder_path (str): Path to the folder containing the annotation data files
    song_id (str): The ID of the song to read data for
    
    Returns:
    np.array: Confusion matrix as a numpy array
    """
    def read_data(file_path):
        """
        Reads the annotation data from a file.
        
        Parameters:
        file_path (str): Path to the file containing the annotation data
        
        Returns:
        list of tuples: List of (start_time, end_time, actual_label, predicted_label)
        """
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                start_time, end_time, labels = line.strip().split()
                actual_label, predicted_label = labels.split(',')
                data.append((float(start_time), float(end_time), int(actual_label), int(predicted_label[1:])))  # Removing 'Q' and converting to int
        return data

    def calculate_time_intervals(data, actual, predicted):
        """
        Calculates the total time intervals for a given actual and predicted label pair.
        
        Parameters:
        data (list of tuples): Annotation data
        actual (int): The actual label to filter
        predicted (int): The predicted label to filter
        
        Returns:
        float: The total time intervals for the given actual and predicted label pair
        """
        total_time = 0
        for start_time, end_time, actual_label, predicted_label in data:
            if actual_label == actual and predicted_label == predicted:
                total_time += (end_time - start_time)
        return total_time

    def confusion_matrix(data):
        """
        Calculates the confusion matrix based on the time intervals.
        
        Parameters:
        data (list of tuples): Annotation data
        
        Returns:
        np.array: A confusion matrix based on time intervals
        list: Categories present in the data
        """
        categories = sorted(set([label for _, _, actual_label, predicted_label in data for label in [actual_label, predicted_label]]))
        
        matrix = np.zeros((len(categories), len(categories)))

        for i, actual in enumerate(categories):
            for j, predicted in enumerate(categories):
                matrix[i, j] = calculate_time_intervals(data, actual, predicted)

        return matrix, categories

    if flag == 0:
        # Read data from the file
        file_path = os.path.join(folder_path, song_id)
        data = read_data(file_path)
    else:
        file_path = os.path.join(folder_path)
        data = read_data(file_path)
    
    # Calculate the confusion matrix
    matrix, categories = confusion_matrix(data)
    
    # Return the confusion matrix
    return matrix

def gather_all_data(folder_path, output_file_path):
    """
    Reads all the .txt files in the specified folder and gathers their contents.
    Saves the gathered data to a specified file.
    Parameters:
    folder_path (str): Path to the folder containing the .txt files
    
    Returns:
    list: A list containing all the lines from all files
    """
    all_data = []
    
    for filename in sorted(os.listdir(folder_path), key=str.lower):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                all_data.extend(file.readlines())

    with open(output_file_path, 'w') as file:
        file.writelines(all_data)


def filter_outliers(file_path, y_pred, threshold=1.0):
    def read_data(file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():
                    start_time, end_time, labels = line.strip().split()
                    quadrant_real, quadrant_pred = labels.split(',')
                    data.append((float(start_time), float(end_time), int(quadrant_real), int(quadrant_pred[1:])))
        return data

    data = read_data(file_path)
    
    for i in range(1, len(data) - 1):
        start_time, end_time, quadrant_real, quadrant_pred = data[i]
        duration = end_time - start_time
        
        if duration < threshold:
            prev_quadrant_pred = data[i - 1][3]
            next_quadrant_pred = data[i + 1][3]
            
            if quadrant_pred != prev_quadrant_pred and prev_quadrant_pred == next_quadrant_pred:
                y_pred[i] = 'Q' + str(prev_quadrant_pred)

    return y_pred

def main_tvt_eval(file_with_mel_train, file_with_labels_train, file_with_mel_test, file_with_labels_test, mode=None, tuner_type=None):
    windth_time = 48

    # Open file with and load Mel-spectrograms representations of the dataset's samples
    with open(file_with_mel_train + '.pickle', 'rb') as mel_file:
        df_spect_train = pickle.load(mel_file)
    # Open file containing data in the format [sample_id, sample_waveform_representation, labels(quadrants in this case)]
    with open(file_with_labels_train + '.pickle', 'rb') as sounds_file:
        sounds_train = pickle.load(sounds_file)

    with open(file_with_mel_test + '.pickle', 'rb') as mel_file:
        df_spect_test = pickle.load(mel_file)

    ## Sort samples by ID
    sorted_sounds_train = sorted(sounds_train, key=lambda x: x[0].lower())
    sorted_sounds_test = sorted(df_spect_test, key=lambda x: x[0].lower())

    df_spect_train_sorted =  sorted(df_spect_train, key=lambda x: x[0].lower())
    df_spect_test_sorted = sorted(df_spect_test, key=lambda x: x[0].lower())
    ## Get the labels/targets for the samples.
    labels_train = np.array([temp[2] for temp in sorted_sounds_train])
    labels_test = np.array([np.array(temp[2]) for temp in sorted_sounds_test])

    """for i, entry in enumerate(df_spect_test_sorted):
        if len(entry) > 1 and isinstance(entry[1], list):
            num_arrays = len(entry[1])
            print(f"Entry {i+1} has {num_arrays} arrays.")

    # If you want to see the total count of arrays across all entries
    total_arrays = sum(len(entry[1]) for entry in df_spect_test_sorted if len(entry) > 1 and isinstance(entry[1], list))
    print(f"Total number of arrays across all entries: {total_arrays}")"""

    song_names_array = [x[0] for x in sorted_sounds_test]
    
    ## Prepare train, validation and test sets
    print('Loading TVT split...')

    # Failsafe
    tvt_files = None

    # Open chosen split strategy
    if '70_15_15' in EVAL_STRATEGY_PATH:
        tvt_files = [CURR_PATH + PATH_TO_DATASET + TVT_70_15_15_PATH + f for f in listdir(CURR_PATH + PATH_TO_DATASET + TVT_70_15_15_PATH) if isfile(join(CURR_PATH + PATH_TO_DATASET + TVT_70_15_15_PATH + f))]
    elif '40_30_30' in EVAL_STRATEGY_PATH:
        tvt_files = [CURR_PATH + PATH_TO_DATASET + TVT_40_30_30_PATH + f for f in listdir(CURR_PATH + PATH_TO_DATASET + TVT_40_30_30_PATH) if isfile(join(CURR_PATH + PATH_TO_DATASET + TVT_40_30_30_PATH + f))]

    # Ensure that one split was chosen
    assert tvt_files is not None

    # Reshape the previously loaded samples and split into corresponding sets.
    df_spect_array_train = np.array(df_spect_train_sorted)
    df_spect_array_test = np.array(df_spect_test_sorted)

    #X_train = np.array([np.array(entry[1]).reshape((len(entry[1]) ,windth_time, 128, 1)) for entry in df_spect_array_train])
    X_train = np.array([np.array(entry[1]).reshape((len(entry[1]) ,windth_time, 128, 1)) for entry in df_spect_array_train])

    X_test = []
    for entry in df_spect_array_test:
        segments = np.array(entry[1])  # Convert to numpy array
        song_data = []
        for segment in segments:
            mini_segments = np.array(segment)  # Convert each segment to numpy array
            segment_data = []
            for mini_segment in mini_segments:
                if mini_segment.shape == (windth_time, 128):
                    reshaped_mini_segment = mini_segment.reshape((windth_time, 128, 1))
                    segment_data.append(reshaped_mini_segment)
                else:
                    print(f"Cannot reshape mini-segment of shape {mini_segment.shape} with size {mini_segment.size} into ({windth_time}, 128, 1)")
            song_data.append(np.array(segment_data))
        X_test.append(np.array(song_data, dtype=object))

    X_test = np.array(X_test, dtype=object)
    # Translate quadrants to ints.
    classes = ['Q1', 'Q2', 'Q3', 'Q4']

    unique_labels = np.unique(labels_train)
    
    # Create a mapping dictionary to map labels to integer values starting from 1
    mapping = {}
    for i, label in enumerate(unique_labels):
        mapping[label] = i + 1  # Increment by 1 to start from 1 instead of 0


    #Map labels to their corresponding integer values using the mapping dictionary

    Y_total_train = [mapping[label] for label in labels_train]
    
    # Transform to one-hot encoding, I think.
    Y_total_train_np = np.array(Y_total_train)
    Y_total_train_adjusted = Y_total_train_np - 1
    Y_train_one_hot = to_categorical(Y_total_train_adjusted, num_classes=4)
    Y_total_train = Y_train_one_hot

    binary_labels_test = []

    for label_array in labels_test:
        binary_label_array = []
        for label in label_array:
            binary_label = np.zeros(4)  # Create a zero vector of length 4
            binary_label[label - 1] = 1  # Set the corresponding index to 1
            binary_label_array.append(binary_label)
        
        # Convert binary_label_array to numpy array
        binary_label_array = np.array(binary_label_array)
        binary_labels_test.append(binary_label_array)

    # Convert binary_labels_test to numpy array
    Y_total_test = np.array(binary_labels_test)
    
    #y_train, y_validate, y_test = Y_total[train_split], Y_total[validation_split], Y_total[test_split]
    y_train = Y_total_train
    y_test = Y_total_test
    # Mudar Ver com pedro
    learning_rate = 0.0001
    epochs = 200
    batch_size = 128
  
    print("Preprocessing complete!")

    print("X_train: {}".format(X_train.shape))
    print("y_train: {}".format(Y_total_train.shape))
    print("X_test: {}".format(X_test.shape))
    print("y_test: {}".format(Y_total_test.shape))

    # Start optimization/evaluation phases timer.
    start = time.time()

    if mode == 'evaluate_all_trials':
        # MUDA ISTO DEPOIS PFF
        run_f1score_tvt_split(X_train, y_train, X_test, y_test, epochs, batch_size, learning_rate, song_names_array)
        
    else:
        print('Not an option!')
        exit(-1)

    # End optimization/evaluation phases timer and calculate elapsed time.
    end = time.time()
    elapsed = round(end - start)
    print("Minutes elapsed: " + str(elapsed / 60))

if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)

    #1.5 seconds approach
   
    #file_with_samples_train = 'CNNTrainSet/merge_audio_complete_normalized_16kHz'
    #file_with_mel_train = '/welmo-data/tomasferreira/CNN_Data/CNNTrainSet/merge_audio_complete_16kHz_melspect_1_5s'

    #file_with_samples_train = 'CNN_Data/CNNTrainSet/merge_audio_balanced_normalized_16kHz'
    #file_with_mel_train = '/welmo-data/tomasferreira/CNN_Data/CNNTrainSet/merge_audio_balanced_16kHz_melspect_1_5s'

    #file_with_samples_test = 'CNN_Data/CNNTestSet/merge_audio_complete_test_normalized_16kHz'
    #file_with_mel_test =  'CNN_Data/CNNTestSet/merge_audio_complete_test_16kHz_melspect_1_5s'

    # allin1 approach

    file_with_samples_train = 'CNNTrainSet/merge_audio_complete_normalized_16kHz'
    file_with_mel_train = 'CNNTrainSet/merge_audio_complete_16kHz_melspect_1_5s'

    file_with_samples_test = 'CNNTestSet/merge_audio_complete_allin1_test_normalized_16kHz'
    file_with_mel_test =  'CNNTestSet/merge_audio_complete_allin1_test_16kHz_melspect4'


    
            # Set which device to use for computations
        # '' -> CPU; '0' -> GPU:0 ...
    #environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    #environ['CUDA_VISIBLE_DEVICES'] = "0"

  
        # Uncomment if you need to generate the samples and mel files
        #load2mel_fixed(True, True, file_with_samples)
        #create_melspectrograms(file_with_samples, file_with_mel)

        # Check your GPUs
        #print(get_available_gpus())

    eval_strat = input("Proceed with:\n1) Cross-validation 10-Fold 10-Reps (Don't choose this one!)\n2) Train-Validation-Test 70-15-15\n3) Train-Validation-Test 40-30-30\n-> ")
    if eval_strat == '1':
        EVAL_STRATEGY_PATH = 'CV_10_Fold_10_Reps/'
    elif eval_strat == '2':
        EVAL_STRATEGY_PATH = 'TVT_70_15_15/'
    elif eval_strat == '3':
        EVAL_STRATEGY_PATH = 'TVT_40_30_30/'

    if '70_15_15' in EVAL_STRATEGY_PATH:
        procedure = input('Proceed with train-test-validation strategy (70-15-15)? -> (Y)es or (N)o\n-> ')

        if procedure == 'Y':
            opt_procedure = input('Do you wish to optimize or evaluate a model, or use predefined hyperparameters? (second option only available after running first) -> (O)ptimize, (E)valuate or (R)un Fixed Params\n-> ')

            if opt_procedure == 'E':
                main_tvt_eval(file_with_mel_train,file_with_samples_train ,file_with_mel_test, file_with_samples_test, mode='evaluate_all_trials', tuner_type='no_tuner')
                print('Done!')
                exit(0)                   

            if opt_procedure == 'R':
                #main_tvt_eval(file_with_mel, file_with_samples)
                print('Under construction...')
                exit(0)
    print('Shutting down...')
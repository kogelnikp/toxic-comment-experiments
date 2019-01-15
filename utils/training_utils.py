import numpy as np
import keras.backend as K
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score
from keras.callbacks import ModelCheckpoint
from keras.models import clone_model
from utils.keras_utils import RocAucEvaluation, DataSet


def train_model(model, X, Y, validation_data, num_epochs, batch_size, optimizer, loss, metrics, weights_path, seed=2018):
    """[summary]
    
    Arguments:
        model -- the keras model to be trained
        X -- training data
        Y -- training labels
        validation_data {[type]} -- validation data and labels
        num_epochs {int} -- number of epochs
        batch_size {int} -- batch size
        optimizer {string} -- optimizer to be used for training
        loss {string} -- loss function to be used for training
        metrics {array of metrics} -- metrics to be calculated during training
    
    Keyword Arguments:
        seed {int} -- seed for random variables (default: {2018})
    
    Returns:
        model -- the trained keras model
        
    """    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    roc_callback_train = RocAucEvaluation((X, Y), output_prefix='train')
    roc_callback_val = RocAucEvaluation(validation_data, output_prefix='val')

    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')

    model.fit(X, Y, validation_data=validation_data, epochs=num_epochs, batch_size=batch_size, \
        callbacks=[checkpoint, roc_callback_train, roc_callback_val])

    #score = model.evaluate(validation_data[0], validation_data[1], batch_size=batch_size)
    predictions = model.predict(validation_data[0], batch_size=batch_size)

    return model, predictions


def train_and_evaluate_model(model_template, X, Y, validation_data, num_epochs, batch_size, optimizer, loss, metrics, weights_path=None, write_model_checkpoints=False, runs=10, seed=2018, dataset=DataSet.KAGGLE):
    """trains and evaluates the given model for the specified number of runs
    and returns the scores in the following format:
    format: [run, train/val, epoch, class, metric]
    train/val:  0..train, 1..val
    class:      0..avg, 1-6..classes
    metric:     0..rocauc, 1..f1
    
    Arguments:
        model {keras model} -- the keras model to be evaluated
        X {array} -- the tokenized comments
        Y {array} -- the labels for each comment
        validation_data {array} -- (X, y) data for validation
        num_epochs {int} -- number of epochs per run
        batch_size {int} -- the batch size
        optimizer {string} -- optimizing algorithm
        loss {string} -- loss function
        metrics {array} -- metrics calculated after every epoch
    
    Keyword Arguments:
        weights_path {string} -- path where the weights should be saved (default: {None})
        write_model_checkpoints {bool} -- should the best model of each run be saved (default: {False})
        runs {int} -- how many runs for evaluation (default: {10})
        seed {int} -- seed for random variables (default: {2018})
    
    Returns:
        numpy array -- train and test scores for each run and epoch
    """
    
    scores = np.zeros((runs, 2, num_epochs, Y.shape[1]+1, 2))

    for run in range(runs):
        print("RUN {}/{}".format(run+1, runs))
        model = clone_model(model_template)
        try:
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            callbacks = []
            roc_callback_train = RocAucEvaluation((X, Y), output_prefix='train', dataset=dataset)
            roc_callback_val = RocAucEvaluation(validation_data, output_prefix='val', dataset=dataset)
            callbacks.append(roc_callback_train)
            callbacks.append(roc_callback_val)

            if write_model_checkpoints:
                callbacks.append(ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min'))
            
            model.fit(X, Y, validation_data=validation_data, epochs=num_epochs, batch_size=batch_size, \
                callbacks=callbacks)
            
            scores[run, 0, :, :, :] = np.array(roc_callback_train.scores)
            scores[run, 1, :, :, :] = np.array(roc_callback_val.scores)

        finally:
            del model
            tf.reset_default_graph()
            K.clear_session()

    return scores
        



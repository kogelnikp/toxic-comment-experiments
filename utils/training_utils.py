import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score
from keras.models import clone_model
from utils.keras_utils import RocAucEvaluation


def train_model(model, X, Y, validation_data, num_epochs, batch_size, optimizer, loss, metrics, seed=2018):
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
    model.fit(X, Y, validation_data=validation_data, epochs=num_epochs, batch_size=batch_size, \
        callbacks=[roc_callback_train, roc_callback_val])

    score = model.evaluate(validation_data[0], validation_data[1], batch_size=batch_size)
    predictions = model.predict(validation_data[0], batch_size=batch_size)

    return model, predictions

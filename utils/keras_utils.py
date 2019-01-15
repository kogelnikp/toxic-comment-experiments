import numpy as np
from enum import Enum
from sklearn.metrics import roc_auc_score, f1_score
from keras.callbacks import Callback

class DataSet(Enum):
    KAGGLE = 0,
    YOUTOXIC = 1


class RocAucEvaluation(Callback):
    """Keras callback class for ROC AUC evaluation.
    Based on: https://www.kaggle.com/yekenot/pooled-gru-fasttext    
    The scores can be retrieved from self.scores where it is stored in the following format:
          | ROC AUC | F1      |
    Avg   |         |         |
    STox  |         |         |
    Tox   |         |         |
    Obs   |         |         |
    Thr   |         |         |
    Ins   |         |         |
    IdH   |         |         |

    """

    def __init__(self, validation_data=(), output_prefix='val', interval=1, dataset=DataSet.KAGGLE):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.output_prefix = output_prefix
        self.scores = []
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            roc_avg = roc_auc_score(self.y_val, y_pred, average='micro')
            roc_cl = roc_auc_score(self.y_val, y_pred, average=None)
            f1_avg = f1_score(self.y_val, y_pred >= 0.5, average='micro')
            f1_cl = f1_score(self.y_val, y_pred >= 0.5, average=None)

            if self.dataset.value == DataSet.KAGGLE.value:
                # print scores
                print("\n {}: ROC-AUC - epoch: {:d} - score: {:.5f}".format(self.output_prefix, epoch+1, roc_avg))
                print(" Tox: {:.5f} - STox: {:.5f} - Obs: {:.5f} - Thr: {:.5f} - Ins: {:.5f} - IdH: {:.5f}".format(roc_cl[0], roc_cl[1], roc_cl[2], roc_cl[3], roc_cl[4], roc_cl[5]))
                print(" {}: F1 Score - epoch: {:d} - score: {:.5f}".format(self.output_prefix, epoch+1, f1_avg))
                print(" Tox: {:.5f} - STox: {:.5f} - Obs: {:.5f} - Thr: {:.5f} - Ins: {:.5f} - IdH: {:.5f}".format(f1_cl[0], f1_cl[1], f1_cl[2], f1_cl[3], f1_cl[4], f1_cl[5]))

                epoch_scores = np.zeros((7, 2))
            elif self.dataset.value == DataSet.YOUTOXIC.value:
                # print scores
                print("\n {}: ROC-AUC - epoch: {:d} - score: {:.5f}".format(self.output_prefix, epoch+1, roc_avg))
                print(" IsToxic: {:.5f} - IsAbusive: {:.5f} - IsHatespeech: {:.5f}".format(roc_cl[0], roc_cl[1], roc_cl[2]))
                print(" {}: F1 Score - epoch: {:d} - score: {:.5f}".format(self.output_prefix, epoch+1, f1_avg))
                print(" sToxic: {:.5f} - IsAbusive: {:.5f} - IsHatespeech: {:.5f}".format(f1_cl[0], f1_cl[1], f1_cl[2]))

                epoch_scores = np.zeros((4, 2))
            
            # save scores
            epoch_scores[0, 0] = roc_avg
            epoch_scores[1:, 0] = roc_cl
            epoch_scores[0, 1] = f1_avg
            epoch_scores[1:, 1] = f1_cl
            self.scores.append(epoch_scores)


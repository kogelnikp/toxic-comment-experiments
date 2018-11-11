from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

class RocAucEvaluation(Callback):
    """Keras callback class for ROC AUC evaluation.
    Taken from: https://www.kaggle.com/yekenot/pooled-gru-fasttext    
    """

    def __init__(self, validation_data=(), output_prefix='val', interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.output_prefix = output_prefix

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            s_cl = roc_auc_score(self.y_val, y_pred, average=None)
            print("\n {}: ROC-AUC - epoch: {:d} - score: {:.6f}".format(self.output_prefix, epoch+1, score))
            print(" Tox: {} - STox: {} - Obs: {} - Thr: {} - Ins: {} - IdH: {}".format(s_cl[0], s_cl[1], s_cl[2], s_cl[3], s_cl[4], s_cl[5]))            

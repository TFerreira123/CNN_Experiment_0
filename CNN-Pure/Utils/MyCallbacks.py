import tensorflow as tf


class MyCallbacks(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyCallbacks, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        acc = logs["accuracy"]
        if acc >= self.threshold:
            self.model.stop_training = True

class MyEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience, metric):
        super(MyEarlyStopping, self).__init__()
        self.initial_patience = patience
        self.patience = patience
        self.metric = metric
        self.best = 9999
    
    def on_epoch_end(self, logs=None):
        metric_value = logs[self.metric]

        if metric_value < self.lowest_value:
            self.best = metric_value
            self.patience = self.initial_patience
            print('Patience restored')
        else:
            self.patience -= 1
            print('Patience at {}'.format(self.patience))
            if self.patience == 0:
                self.model.stop_training = True
            
        


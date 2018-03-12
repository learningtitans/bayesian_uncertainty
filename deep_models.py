from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import scipy.stats

from keras.layers.core import Lambda
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate
from keras.optimizers import Adam

from tensorflow.contrib.distributions import Normal

BATCH_SIZE = 1000
N_EPOCHS = 50
BD_SAMPLES = 50


class MLPBayesianDropout(BaseEstimator, RegressorMixin):  
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.model = Sequential()
        self.model.add(Dense(512, activation='relu', input_shape=(X.shape[-1],)))
        self.model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
        self.model.add(Dense(1, activation='linear'))

        self.model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['mse'])

        self.model.fit(X, y,
                    batch_size=BATCH_SIZE,
                    epochs=N_EPOCHS,
                    verbose=0)
        
        return self

    def predict(self, X, y=None):
        pred = [self.model.predict(X).reshape(-1) for _ in range(BD_SAMPLES)]
        pred_mean = np.mean(pred, axis=0)
        pred_std = np.std(pred, axis=0)
        return pred_mean, pred_std
    
    
class MLPNormal(BaseEstimator, RegressorMixin):  
    def __init__(self):
        pass
    
    def fit(self, X, y):
        input_ = Input(shape=(X.shape[-1],))
        x = Dense(X.shape[-1], activation='relu')(input_)
        #x = Dropout(0.5)(x)
        #x = Dense(512, activation='relu')(x)
        #x = Dropout(0.5)(x)
        mean = Dense(1, activation='linear')(x)
        std = Dense(1, activation=lambda y: K.exp(y))(x)
        #std = Dense(1, activation='linear')(x)
        out = concatenate([mean, std])

        self.model = Model(inputs=[input_], outputs=[out])

        def normalLL(yTrue, yPred):
            mean = yPred[:,0]
            std = yPred[:,1]+1e-5
            return -K.mean(Normal(mean, std).log_prob(yTrue))

#         def log_gaussian2(x, mean, log_std):
#             log_var = 2*log_std
#             return -np.log(2*np.pi)/2.0 - log_var/2.0 - (x-mean)**2/(2*K.exp(log_var))

#         def normalLL(y_true, y_pred):
#             return -K.mean(log_gaussian2(y_pred[:, 0], y_true[:, 0], y_pred[:, 1]))

        self.model.compile(loss=normalLL,
              optimizer=Adam(lr=1e-4),
              metrics=['mse'])

        self.model.fit(X, y,
                    batch_size=BATCH_SIZE,
                    epochs=N_EPOCHS,
                    verbose=1)
        
        return self

    def predict(self, X, y=None):
        pred_mean = self.model.predict(X)[:,0]
        pred_std = self.model.predict(X)[:,1]+1e-5
        #pred_std = np.exp(self.model.predict(X)[:,1])
        print(np.mean(pred_mean), np.mean(pred_std))
        return pred_mean, pred_std
    
    
class MLPBaseline(BaseEstimator, RegressorMixin):  
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.model = Sequential()
        self.model.add(Dense(512, activation='relu', input_shape=(X.shape[-1],)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='linear'))

        self.model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['mse'])

        self.model.fit(X, y,
                    batch_size=BATCH_SIZE,
                    epochs=N_EPOCHS,
                    verbose=0)
        
        errors = y - self.model.predict(X).reshape(-1)
        self.std = np.std(errors)
        
        return self

    def predict(self, X, y=None):
        pred_mean = self.model.predict(X).reshape(-1) 
        pred_std = self.std * np.ones(len(pred_mean))
        return pred_mean, pred_std
    
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import scipy.stats

from keras.layers.core import Lambda
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate
from keras.optimizers import Adam
from keras.regularizers import l2

from tensorflow.contrib.distributions import Normal

BATCH_SIZE = 256
N_EPOCHS = 40
BD_SAMPLES = 1000
N_HIDDEN = 50
DROPOUT = 0.05

def l2_reg(N):
    tau = 192.482910156 # obtained from https://github.com/yaringal/DropoutUncertaintyExps
    lengthscale = 1e-2
    return lengthscale**2 * (1 - DROPOUT) / (2. * N * tau)
            
    
class MLPBayesianDropout(BaseEstimator, RegressorMixin):  
    def __init__(self):
        pass
    
    def fit(self, X, y):        
        reg = l2_reg(X.shape[0])
                               
        self.model = Sequential()
        self.model.add(Dense(N_HIDDEN, activation='relu', kernel_regularizer=l2(reg), input_shape=(X.shape[-1],)))
        self.model.add(Lambda(lambda x: K.dropout(x, level=DROPOUT)))
        self.model.add(Dense(N_HIDDEN, activation='relu', kernel_regularizer=l2(reg)))
        self.model.add(Lambda(lambda x: K.dropout(x, level=DROPOUT)))
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
        reg = l2_reg(X.shape[0])

        input_ = Input(shape=(X.shape[-1],))
        x = Dense(N_HIDDEN, activation='relu', kernel_regularizer=l2(reg),)(input_)
        x = Dropout(DROPOUT)(x)
        x = Dense(N_HIDDEN, activation='relu', kernel_regularizer=l2(reg),)(x)
        x = Dropout(DROPOUT)(x)
        mean = Dense(1, activation='linear')(x)
        std = Dense(1, activation=lambda y: K.exp(y))(x)
        out = concatenate([mean, std])

        self.model = Model(inputs=[input_], outputs=[out])

        def normalLL(yTrue, yPred):
            mean = yPred[:,0]
            std = yPred[:,1]
            return -K.mean(Normal(mean, std).log_prob(yTrue))

        self.model.compile(loss=normalLL,
              optimizer=Adam(),
              metrics=['mse'])

        self.model.fit(X, y,
                    batch_size=BATCH_SIZE,
                    epochs=N_EPOCHS,
                    verbose=0)
        
        return self

    def predict(self, X, y=None):
        pred_mean = self.model.predict(X)[:,0]
        pred_std = self.model.predict(X)[:,1]
        return pred_mean, pred_std
    
    
class MLPBaseline(BaseEstimator, RegressorMixin):  
    def __init__(self):
        pass
    
    def fit(self, X, y):
        reg = l2_reg(X.shape[0])

        self.model = Sequential()
        self.model.add(Dense(N_HIDDEN, activation='relu', kernel_regularizer=l2(reg), input_shape=(X.shape[-1],)))
        self.model.add(Dropout(DROPOUT))
        self.model.add(Dense(N_HIDDEN, activation='relu', kernel_regularizer=l2(reg)))
        self.model.add(Dropout(DROPOUT))
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
    
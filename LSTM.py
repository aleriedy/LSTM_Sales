
# coding: utf-8

# # PARAMETERS 

# In[1]:

janela = 6 #tamanho da Janela deslizante

problem_name = 'lstm_sales' #to save the models
model_architecture = 'VGG_16'
weights_path = None 
target_size = (224, 224) 
batch_size = 1

epochs = 100 #após x épocas sem melhorar pará (a usar callback)


# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math, time
import datetime
from keras.models import Sequential
from keras.layers.recurrent import LSTM
# fixar random seed para se puder reproduzir os resultados
seed = 9
np.random.seed(seed)

from keras.models import load_model
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding####################
from keras.preprocessing import sequence
from keras.constraints import maxnorm 
from keras.optimizers import SGD 
from keras.utils import np_utils 
from keras import backend as K 
from keras.utils.data_utils import get_file
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import keras
K.set_image_dim_ordering('tf') #ordem 'th' ou 'tf' 
from numpy import genfromtxt
import math 

from timeit import default_timer as timer
from time import time as tick
import matplotlib.pyplot as plt 
import pickle 
from os import listdir
from PIL import Image, ImageOps
from os.path import isfile, join
import os
from scipy.misc	import toimage 
from scipy import misc, ndimage
import scipy.fftpack as pack
import scipy.misc
from scipy.ndimage import rotate
from sklearn.datasets import fetch_mldata
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
import pathlib
import datetime


# In[3]:

#função load_data do lstm.py configurada para aceitar qualquer número de parametros
#o último atributo é que fica como label (resultado)
#stock é um dataframe do pandas (uma especie de dicionario + matriz)
#seq_len é o tamanho da janela a ser utilizada na serie temporal
def load_data(df_dados, janela):
    qt_atributos = len(df_dados.columns)
    mat_dados = df_dados.as_matrix() #converter dataframe para matriz (lista com lista de cada registo)
    tam_sequencia = janela + 1
    res = []
    for i in range(len(mat_dados) - janela): #numero de registos - tamanho da sequencia
        res.append(mat_dados[i: i + tam_sequencia])
    
    res = np.array(res) #dá como resultado um np com uma lista de matrizes (janela deslizante ao longo da serie)

    #qt_casos_treino = int(round(0.9 * res.shape[0])) #90% passam a ser casos de treino
    
    qt_casos_treino = 24 # 2 anos
    #qt_casos_test = 12 - janela
    
    x_train = res[:qt_casos_treino, :-1] #menos um registo pois o ultimo registo é o registo a seguir à janela
    y_train = res[:qt_casos_treino, -1][:,-1] #para ir buscar o último atributo para a lista dos labels
    x_test = res[qt_casos_treino:, :-1]
    y_test = res[qt_casos_treino:, -1][:,-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], qt_atributos))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], qt_atributos))
    return [x_train, y_train, x_test, y_test]

#imprime um grafico com os valores de teste e com as correspondentes tabela de previsões
def print_series_prediction(y_test,predic):
    diff=[]
    racio=[]
    for i in range(len(y_test)): #para imprimir tabela de previsoes
        racio.append( (y_test[i]/predic[i])-1)
        diff.append( abs(y_test[i]- predic[i]))
        print('valor: %f ---> Previsão: %f Diff: %f Racio: %f' % (y_test[i], predic[i], diff[i], racio[i]))
    plt.plot(y_test,color='blue', label='y_test')
    plt.plot(predic,color='red', label='prediction') #este deu uma linha em branco
    plt.plot(diff,color='green', label='diff')
    plt.plot(racio,color='yellow', label='racio')
    plt.legend(loc='upper left')
    plt.show()
    
    


def print_model(model,fich):
    from keras.utils import plot_model
    plot_model(model, to_file=fich, show_shapes=True, show_layer_names=True)
    
def print_history_accuracy(history):
    print(history.history.keys())
    plt.plot(history.history['mean_squared_error'])
    plt.title('mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def print_history_loss(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

def load_batch(fpath, label_key='labels'): 
 
    f = open(fpath, 'rb') 
    d = pickle.load(f, encoding='bytes') 
    d_decoded = {}        # decode utf8 
    for k, v in d.items(): 
        d_decoded[k.decode('utf8')] = v 
    d = d_decoded 
    f.close() 
    data = d['data'] 
    labels = d[label_key] 
    data = data.reshape(data.shape[0], 3, 32, 32) 
    return data, labels


def rotate_resize(temp, tam_image):
    #-------------------rodar se necessário e cortar em quadrado
    if temp.shape[0] > temp.shape[1]:
        temp = rotate(temp,90)
    
    #cortar em quadrado no centro da imagem e fazer resize para o tam_image
    difShapes = temp.shape[1]-temp.shape[0]
    return (255 * resize(temp[0:temp.shape[0],int(difShapes/2):int(difShapes/2)+temp.shape[0]],
                            (tam_image, tam_image))).astype(np.uint8)


# In[4]:

def read_and_pre_process():
    file_name = 'advertising-and-sales-data-36-co.csv'
    col_names = 'date', 'pub', 'sales'
    dataset = pd.read_csv(file_name, sep = ';', header=0, names=col_names) #3 colunas
    df = pd.DataFrame(dataset)
    date_split = df['date'].str.split('-').str
    df['year'], df['month'] = date_split #acrescentar ano e mes separados
    df.drop(df.columns[[0]], axis=1, inplace=True) #eliminar data original

    df = df[:-1] #eliminar a ultima linha porque é uma frase informativa

    #vamos passar ano e mes para para strings para não ser interpretado como valores

    look_up = {'1': 'First', '2': 'Second', '3': 'Third'}
    #df['year'] = df['year'].apply(lambda x: look_up[x])

    look_up = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May',
                '06': 'Jun', '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}

    #df['month'] = df['month'].apply(lambda x: look_up[x])

    df = df[['year', 'month', 'pub', 'sales']]
    return df


# # Models

# In[5]:

def build_model7(janela, nmr_parametros):
    #embedding_vecor_length = 32
    
    model = Sequential()
    
    model.add(BatchNormalization(input_shape=(janela, nmr_parametros)))
    model.add(LSTM(256, input_shape=(janela, nmr_parametros), return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(BatchNormalization(input_shape=(janela, nmr_parametros)))
    model.add(LSTM(128, input_shape=(janela, nmr_parametros), return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(BatchNormalization(input_shape=(janela, nmr_parametros)))
    model.add(LSTM(64, input_shape=(janela, nmr_parametros), return_sequences=False))
    model.add(Dropout(0.5))
    
    model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
    model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['mse'])
    return model


# # Callbacks

# In[6]:

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

history_loss = LossHistory() #print(history.losses) to use      

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min')    

checkpoint = ModelCheckpoint(filepath = 'checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5', monitor='mean_squared_error', save_best_only=True, mode='min', period=1)

#reduce training rate when no improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

csv_logger = CSVLogger('training.log')


# # Training

# In[13]:

if __name__ == '__main__':
    df = read_and_pre_process()
    print(df)
    X_train, y_train, X_test, y_test = load_data(df, janela)# o df[::-1] é o df por ordem inversa

    #max_review_length = 10
    #x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
    #x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)
    
    nmr_parametros = X_train.shape[2]
    
    model = build_model7(janela, nmr_parametros)
    print(model.summary())
    history = model.fit(X_train, y_train, batch_size=512, epochs=1000, validation_split=0.1, verbose=1, callbacks = [checkpoint])
    
    print_history_accuracy(history) 
    
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
    
    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    
    print(model.metrics_names)
    p = model.predict(X_test)
    predic = np.squeeze(np.asarray(p)) #transformar uma matriz de uma coluna e n linhas em um np array de n elementos
    print_series_prediction(y_test,predic)
    
    
    ''' 
    MSE- (Mean square error), RMSE- (root mean square error) –
    o significado de RMSE depende do range da label. para o mesmo range menor é melhor.
    '''


# In[ ]:



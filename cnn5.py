import numpy as np
from keras import models
from keras import layers
from keras import optimizers,initializers
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
from keras.models import load_model

def remove_zero_rows(X,Y):
    # X is a scipy sparse matrix. We want to remove all zero rows from it
    nonzero_row_indice, _ = X.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)
    return X[unique_nonzero_indice],Y[unique_nonzero_indice]

def read_data(filenamex,filenamey):

    x = np.load(filenamex)
    y = np.load(filenamey)
    x,y = remove_zero_rows(x,y)
    size = np.shape(x)[0]
    boards = np.zeros([size,4,4,16])
    directions = np.zeros([size,4])
    for i in range(size):
        for j in range(16):
            num = x[i,j]
            if num==0:
                boards[i][int(j/4)][j%4][0] = 1
            else:
                boards[i][int(j/4)][j%4][int(np.log2(num))] = 1
        directions[i][y[i]] = 1
    
    return boards, directions

NUM_EPOCHS = 1
NUM_CLASSES = 4                                             # four directions
BATCH_SIZE = 1000
INPUT_SHAPE = (4, 4, 16)                                    # from 0 to 2048

# model = load_model('model_best_cnn3_6550.h5')
filepath='model_best_cnn5_6901.h5'
# model_best_cnn3_6520.h5 2 more epochs
# 02 550.4
# 

model = models.Sequential()
model.add(layers.Conv2D(256,(2,2),activation='relu',padding='same',
    kernel_initializer=initializers.random_normal(stddev=0.1),bias_initializer=initializers.Constant(value=0.1),input_shape=INPUT_SHAPE))
model.add(layers.Conv2D(256,(2,2),activation='relu',padding='same',
    kernel_initializer=initializers.random_normal(stddev=0.1),bias_initializer=initializers.Constant(value=0.1)))
model.add(layers.Conv2D(256,(2,2),activation='relu',padding='same',
    kernel_initializer=initializers.random_normal(stddev=0.1),bias_initializer=initializers.Constant(value=0.1)))
model.add(layers.Conv2D(256,(2,2),activation='relu',padding='same',
    kernel_initializer=initializers.random_normal(stddev=0.1),bias_initializer=initializers.Constant(value=0.1)))
model.add(layers.Conv2D(256,(2,2),activation='relu',padding='same',
    kernel_initializer=initializers.random_normal(stddev=0.1),bias_initializer=initializers.Constant(value=0.1)))

model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))

model.add(layers.Dense(NUM_CLASSES,activation='softmax'))

#earlystopping=EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='auto')
#checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='auto')  # auto淇濆瓨,max鍙繚瀛樻渶浣?
#callbacks_list = [checkpoint]

model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
              loss='categorical_crossentropy',metrics=['accuracy'])

for i in range(8):

    # if i < 10: fn = "./TRdata/shuffle_0" + str(i) + "a.txt"
    # else: fn = "./TRdata/shuffle_" + str(i) + "a.txt"
    #fn = "./TRdata/shuffle_0" + str(i) + "a.txt"
    #fn = "./TRdata/shuffle_" + str(i+50) + "a.txt"
    fnx = "./zhc_data/Xe_"+str(i+35)+".npy"
    fny = "./zhc_data/Ye_"+str(i+35)+".npy"
    (x_train,y_train)=read_data(fnx,fny)
    model.fit(x_train,y_train,epochs=NUM_EPOCHS,batch_size=BATCH_SIZE)
    model.save(filepath)
    #model.fit(x_train,y_train,epochs=NUM_EPOCHS,batch_size=BATCH_SIZE,validation_split=0.1,
    #          callbacks=callbacks_list)
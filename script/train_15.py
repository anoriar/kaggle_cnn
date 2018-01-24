# train Accuracy: 99.46%
# time 180 * 10s
# test kaggle 0.99100
#Layer (type)                 Output Shape              Param #
#conv2d_1 (Conv2D)            (None, 16, 28, 28)        160
#conv2d_2 (Conv2D)            (None, 16, 28, 28)        2320
#dropout_1 (Dropout)          (None, 16, 28, 28)        0
#max_pooling2d_1 (MaxPooling2 (None, 16, 14, 14)        0
#conv2d_3 (Conv2D)            (None, 32, 14, 14)        4640
#conv2d_4 (Conv2D)            (None, 32, 14, 14)        9248
#max_pooling2d_2 (MaxPooling2 (None, 32, 7, 7)          0
#dropout_2 (Dropout)          (None, 32, 7, 7)          0
#conv2d_5 (Conv2D)            (None, 64, 7, 7)          18496
#conv2d_6 (Conv2D)            (None, 64, 7, 7)          36928
#dropout_3 (Dropout)          (None, 64, 7, 7)          0
#global_average_pooling2d_1 ( (None, 64)                0
#dense_1 (Dense)              (None, 500)               32500
#dropout_4 (Dropout)          (None, 500)               0
#dense_2 (Dense)              (None, 10)                5010
# https://www.kaggle.com/c/digit-recognizer/leaderboard

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


np.random.seed(42)

train = pd.read_csv('../input/train.csv')
print(train.shape)
test = pd.read_csv('../input/test.csv')
print(test.shape)

labels = train.iloc[:, 0].values.astype('int32')
X_train = (train.iloc[:, 1:].values).astype('float32')
X_test = (test.values).astype('float32')

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

y_train = np_utils.to_categorical(labels)
num_classes = y_train.shape[1]

X_train = X_train / 255
X_test = X_test / 255

mean = np.std(X_train)
X_train -= mean
X_test -= mean

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(1, 28, 28)))
model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(GlobalAveragePooling2D())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

scores = model.evaluate(X_train, y_train, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=1)


def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1, len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)
write_preds(preds, '../output/predictions.csv')

model_json = model.to_json()
with open('../output/digitrecognizer.model.json', 'w') as json_file:
    json_file.write(model_json)
    json_file.close()

model.save_weights('../output/digitrecognizer.model.best.hdf5')

model.summary()

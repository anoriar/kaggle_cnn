# Accuracy: 99.75%
# time 167*10s
# test(kaggle)  0.98815
#Layer (type)                 Output Shape              Param #
#conv2d_1 (Conv2D)            (None, 32, 26, 26)        320
#conv2d_2 (Conv2D)            (None, 32, 24, 24)        9248
#max_pooling2d_1 (MaxPooling2 (None, 32, 12, 12)        0
#dropout_1 (Dropout)          (None, 32, 12, 12)        0
#flatten_1 (Flatten)          (None, 4608)              0
#dense_1 (Dense)              (None, 128)               589952
#dropout_2 (Dropout)          (None, 128)               0
#dense_2 (Dense)              (None, 10)                1290

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
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
model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(1, 28, 28), dim_ordering='th'))
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
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

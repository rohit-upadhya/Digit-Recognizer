import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator

sns.set(style='white', context='notebook',palette='deep')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1)

del train

g = sns.countplot(Y_train)

Y_train.value_counts()

X_train.isnull().any().describe()

test.isnull().any().describe()

X_train = X_train / 255.0
test = test / 255.0

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


Y_train = to_categorical(Y_train, num_classes = 10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train, test_size = 0.1, random_state = 0)

g = plt.imshow(X_train[1][:,:,0])



classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPool2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPool2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow(X_train,Y_train, batch_size = 32)

test_set = test_datagen.flow(X_val,Y_val, batch_size = 32)

history = classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         epochs = 5,
                         validation_data = test_set)
from keras.datasets import cifar10
import matplotlib.pyplot as plt
 
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

(train_X,train_Y),(test_X,test_Y)=cifar10.load_data()

# n=6
# plt.figure(figsize=(20,10))
# for i in range(n):
#     plt.subplot(330+1+i)
#     plt.imshow(train_X[i])
#     plt.show()
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

train_x=train_X.astype('float32')
test_X=test_X.astype('float32')
 
train_X=train_X/255.0
test_X=test_X/255.0

train_Y=np_utils.to_categorical(train_Y)
test_Y=np_utils.to_categorical(test_Y)
 
num_classes=test_Y.shape[1]

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),
    padding='same',activation='relu',
    kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

sgd=SGD(lr=0.01,momentum=0.9,decay=(0.01/25),nesterov=False)
 
model.compile(loss='categorical_crossentropy',
  optimizer=sgd,
  metrics=['accuracy'])

model.summary()

model.fit(train_X,train_Y,
    validation_data=(test_X,test_Y),
    epochs=1,batch_size=32)

acc=model.evaluate(test_X,test_Y)
print(acc*100)

model.save("image_CL.h5")
loaded_model = tf.keras.models.load_model("image_CL.h5")
results={
   0:'aeroplane',
   1:'automobile',
   2:'bird',
   3:'cat',
   4:'deer',
   5:'dog',
   6:'frog',
   7:'horse',
   8:'ship',
   9:'truck'
}

# im=im.resize((32,32))
# im=np.expand_dims(im,axis=0)
# im=np.array(im)
# pred=model.predict_step([im])[0]
# print(pred,results[pred])
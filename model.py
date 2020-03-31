# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import Sequential
# from tensorflow.keras.models import load_model
from keras.layers import Dense
from tensorflow.keras.models import model_from_json
import numpy as np
from keras.preprocessing.image import img_to_array
import os
import cv2
import random
from imutils import paths
# from imutils import build_montages
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.python.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.optimizers import Adam
from keras import optimizers

IMG_HEIGHT, IMG_WIDTH = 150,150


CLASS_NAMES = ['buildings', 'forest', 'glacier' , 'mountain', 'sea', 'street' ]

def get_images(directory, labels):
    Images = []
    Labels = []  # 0 for Building , 1 for forest, 2 for glacier, 3 for mountain, 4 for Sea , 5 for Street
      
    for labels in labels:
        label = 0
        if labels == 'glacier': #Folder contain Glacier Images get the '2' class label.
            label = 2
        elif labels == 'sea':
            label = 4
        elif labels == 'buildings':
            label = 0
        elif labels == 'forest':
            label = 1
        elif labels == 'street':
            label = 5
        elif labels == 'mountain':
            label = 3
        
    for directory in directory: #Extracting the file name of the image from Class Label folder
        image = cv2.imread(directory) #Reading the image (OpenCV)
        image = cv2.resize(image,(150,150)) #Resize the image, Some images are different sizes. (Resizing is very Important)
        Images.append(image)
        Labels.append(label)
  
    return Images,Labels

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [150, 150])
def get_label(label):
  # convert the path to a list of path components
  parts = tf.constant(label)
  # The second to last is the class-directory
  CLASS_NAMES = ['buildings', 'forest', 'glacier' , 'mountain', 'sea', 'street' ]
  return tf.constant(parts == CLASS_NAMES)


def process_path(file_path):
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img

def mapfn(fn,arr):
    wer = []
    for a in arr:
        wer.append(fn(a))
    return wer
    

class MLModel:
    
    def __init__(self,model_json,model_h5):
        # json_file = open(model_json,'r')
        # self.loaded_model_json = json_file.read()
        
        # json_file.close()
        # loaded_model = model_from_json(self.loaded_model_json)
        # # load weights into new model
        self.model = load_model(model_h5)
        print(self.model)



    def predict_all_images(self,imagePaths):
        # initialize our list of results
        results = []

        # loop over our sampled image paths
        for p in imagePaths:
            # load our original input imag
            orig = cv2.imread(p)

            # pre-process our image by converting it from BGR to RGB channel
            # ordering (since our Keras mdoel was trained on RGB ordering),
            # resize it to 64x64 pixels, and then scale the pixel intensities
            # to the range [0, 1]
            image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (150, 150))
            image = image.astype("float") / 255.0

            # order channel dimensions (channels-first or channels-last)
            # depending on our Keras backend, then add a batch dimension to
            # the image
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # make predictions on the input image
            pred = self.model.predict(image)
            print(pred)
            pd1 = pred.argmax(axis=1)[0]
            
            results.append([p,pd1,np.max(pred[0])])


        return results
    
    def get_label(self,label):
        parts = tf.constant(label)
        CLASS_NAMES = ['buildings', 'forest', 'glacier' , 'mountain', 'sea', 'street' ]
        return CLASS_NAMES[label]



    def parameters(self,learning, loss_function, optimizer, file_path, label,epoch):

        
        if optimizer == 'sgd':
            optimizer = optimizers.SGD(lr=learning, decay=1e-6, momentum=0.9, nesterov=True)
        if optimizer == 'RMSprop':
            optimizer = optimizers.RMSprop(learning_rate=learning, rho=0.9)
        # if optimizer == 'Adagrad':  
        #     optimizer = optimizers.Adagrad(learning_rate=learning)
        # if optimizer == 'Adaleta':   
        #     optimizer = optimizers.Adadelta(learning_rate=learning, rho=0.95)
        if optimizer == 'Adam':    
            optimizer = optimizers.Adam(learning_rate=learning, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # if optimizer == 'Admax':    
        #     optimizer = optimizers.Adamax(learning_rate=learning, beta_1=0.9, beta_2=0.999)  
        # if optimizer == 'Nadam':    
        #     optimizer = optimizers.Nadam(learning_rate=learning, beta_1=0.9, beta_2=0.999)

        # image_batch = map(process_path, file_path)
        # label_batch = map(get_label, label)

        image_batch, label_batch = get_images(file_path,label)


        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        data_dir = "intel-image-classification"
        train_data_gen = image_generator.flow_from_directory(directory=str(data_dir+'/seg_test/seg_test'),
                                                            batch_size=64,
                                                            shuffle=True,
                                                            target_size=(150, 150),
                                                            classes = list(CLASS_NAMES))

        # image_v, label_v = next(iter(train_data_gen)) 


        self.model.compile(optimizer="Adam",
                        loss=loss_function,
                        metrics=['accuracy'])
        
        h = self.model.fit(np.array(image_batch), np.array(label_batch), epochs=int(epoch),batch_size= 2)
        
        return
        # return max(history['accuracy']), max(history['val_accuracy']), history['accuracy'], history['val_accuracy']
    def playground(self, pre_name,learning_rate, loss_function, epochs,  optimizer,working,l1=100 , l2=100 ,l3=100,l4=100,l5=100,k1=3,k2=3,k3=3,k4=3,a1="relu",a2='relu',a3='relu',a4='relu',a5='relu'):
        CLASS_NAMES = ['buildings', 'forest', 'glacier' , 'mountain', 'sea', 'street' ]
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        data_dir = "intel-image-classification"
        

        print(epochs)

        model = self.model  
        train_data_gen = image_generator.flow_from_directory('intel-image-classification/seg_test/seg_test',
                                                        
                                                        batch_size=32,
                                    
                                                        shuffle=True,
                                                        target_size=(150, 150),
                                                        classes = list(CLASS_NAMES))



        print("============================",train_data_gen)
        if pre_name == 'vgg'or 'incep'or 'XCEPT':

            # if working:
            #     # for i,layers in model.layers:
            #     #     if i>=fine_tuning1 and i<=fine_tuning2:
            #     #         layer.trainable = True
            #     #     else:
            #     #         layer.trainable = False
            # else:
            #     # for i,layers in model.layers:
            #     #     if i>=fine_tuning1 and i<=fine_tuning2:
            #     #         layer.trainable = False
            #     #     else:
            #     #         layer.trainable = True
                    
            model.compile(optimizer='adam', loss = tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'] )

            history = model.fit_generator(generator=train_data_gen,
                                    steps_per_epoch=3000/64,
                                    epochs=10
                                    )


        if pre_name == 'divyam':
            model = Models.Sequential()

            model.add(Layers.Conv2D(l1,kernel_size=(k1,k1),activation=a1,input_shape=(150,150,3)))
            model.add(Layers.Conv2D(l2,kernel_size=(k2,k2),activation=a2))
            model.add(Layers.MaxPool2D(5,5))

            model.add(Layers.Conv2D(l3,kernel_size=(k3,k3),activation=a3))
            model.add(Layers.Conv2D(l4,kernel_size=(k4,k4),activation=a4))


            model.add(Layers.MaxPool2D(5,5))
            model.add(Layers.Flatten())
            model.add(Layers.Dense(l5,activation=a5))

            model.add(Layers.Dropout(rate=0.5))
            model.add(Layers.Dense(6,activation='softmax'))

            model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])


            history = model.fit( train_data_gen, 
                                            epochs=epochs,
                                            steps_per_epoch=3000/64)                           



        return 



# model = MLModel("./models/divyam.json","./models/divyam.h5")



# model.playground("divyam",0.0001,"asd",10,"ADAA",False)
# # model.parameters(0.0001,tf.keras.losses.SparseCategoricalCrossentropy(), 'Adam',[
# #     "C:/Users/Nisha/Downloads/DATA/10004.jpg",
#     "C:/Users/Nisha/Downloads/DATA/10004.jpg"
# ],["buildings","buildings"])



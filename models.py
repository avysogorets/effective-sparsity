import tensorflow as tf
from tensorflow.keras import layers,models

def get_model(shape,architecture,**kwargs):
  if architecture.lower()=="lenet300100":
    return Lenet300100(shape=shape,**kwargs).build()
  if architecture.lower()=="lenet5":
    return Lenet5(shape=shape,**kwargs).build()
  if architecture.lower()=="vgg16":
    return VGG16(shape=shape,**kwargs).build()
  if architecture.lower()=="vgg19":
    return VGG19(shape=shape,**kwargs).build()
  if architecture.lower()=="resnet18":
    return Resnet18(shape=shape,**kwargs).build()

class Lenet300100():
    def __init__(self,shape,batchnorm=True,decay=5e-4,output_classes=10,activation=True,**kwargs):
        self.output_classes=output_classes
        self.shape=shape
        self.activation=activation
        self.batchnorm=batchnorm
        self.decay=decay
        self.initializer=tf.keras.initializers.VarianceScaling(scale=2.0,mode="fan_avg",distribution="normal")
    def build(self):
        inputs=layers.Input(shape=self.shape)
        x=layers.Flatten(name="output_0")(inputs)
        x=layers.Dense(units=300,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,activation=None,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_dense_1")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_1")(x)
        x=layers.Dense(units=100,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,activation=None,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_dense_2")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_2")(x)
        x=layers.Dense(units=self.output_classes,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,activation=None,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_dense_3")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("softmax",name="output_3")(x)
        model=models.Model(inputs,x,name="LeNet-300-100")
        tensors=[]
        for layer in range(len(model.layers)):
            if "tensor" in model.layers[layer].name:
                tensors.append(layer)
        return model,tensors

class Lenet5():
    def __init__(self,shape,batchnorm=True,decay=5e-4,output_classes=10,activation=True,**kwargs):
        self.output_classes=output_classes
        self.shape=shape
        self.activation=activation
        self.batchnorm=batchnorm
        self.decay=decay
        self.initializer=tf.keras.initializers.VarianceScaling(scale=2.0,mode="fan_avg",distribution="normal")
    def build(self):
        inputs=layers.Input(shape=self.shape,name="output_0")
        x=layers.Conv2D(filters=6,kernel_size=(5,5),padding="valid",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv1")(inputs)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_1")(x)
        x=layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="valid",name="pool2_1")(x)
        x=layers.Conv2D(filters=16,kernel_size=(5,5),padding="valid",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv2")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_2")(x)
        x=layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="valid",name="pool2_2")(x)
        x=layers.Flatten(name="flatten")(x)
        x=layers.Dense(units=120,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,activation=None,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_dense1")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_3")(x)
        x=layers.Dense(units=84,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,activation=None,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_dense2")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_4")(x)
        x=layers.Dense(units=self.output_classes,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,activation=None,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_dense3")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("softmax",name="output_5")(x)
        model=models.Model(inputs,x,name="LeNet-5")
        tensors=[]
        for layer in range(len(model.layers)):
            if "tensor" in model.layers[layer].name:
                tensors.append(layer)
        return model,tensors

class VGG16():
    def __init__(self,shape,pool="max",batchnorm=False,decay=1e-4,output_classes=10,activation=True,**kwargs):
        self.output_classes=output_classes
        self.shape=shape
        self.pool=pool
        self.activation=activation
        self.decay=decay
        self.batchnorm=batchnorm
        self.initializer=tf.keras.initializers.VarianceScaling(scale=2.0,mode="fan_avg",distribution="normal")
    def build(self):
        inputs=layers.Input(shape=self.shape,name="output_0")
        x=layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv1")(inputs)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_1")(x)
        x=layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv2")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_2")(x)
        if self.pool=="max":
            x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
        else:
            x=layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
        x=layers.Conv2D(filters=128,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv3")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_3")(x)
        x=layers.Conv2D(filters=128,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv4")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_4")(x)
        if self.pool=="max":
            x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
        else:
            x=layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
        x=layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv5")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_5")(x)
        x=layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv6")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_6")(x)
        x=layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv7")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_7")(x)
        if self.pool=="max":
            x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
        else:
            x=layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv8")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_8")(x)
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv9")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_9")(x)
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv10")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_10")(x)
        if self.pool=="max":
            x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
        else:
            x=layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv11")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_11")(x)
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv12")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_12")(x)
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv13")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_13")(x)
        if self.pool=="max":
            x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
        else:
            x=layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
        x=layers.Flatten()(x)
        x=layers.Dense(units=self.output_classes,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,activation=None,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_dense2")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("softmax",name="output_15")(x)
        model=models.Model(inputs,x,name="VGG-16")
        tensors=[]
        for layer in range(len(model.layers)):
            if "tensor" in model.layers[layer].name:
                tensors.append(layer)
        return model,tensors

class VGG19():
    def __init__(self,shape,pool="max",batchnorm=False,decay=1e-4,output_classes=100,activation=True,**kwargs):
        self.output_classes=output_classes
        self.shape=shape
        self.pool=pool
        self.decay=decay
        self.activation=activation
        self.batchnorm=batchnorm
        self.initializer=tf.keras.initializers.VarianceScaling(scale=2.0,mode="fan_avg",distribution="normal")
    def build(self):
        inputs=layers.Input(shape=self.shape,name="output_0")
        x=layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv1")(inputs)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_1")(x)
        x=layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv2")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_2")(x)
        x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
        x=layers.Conv2D(filters=128,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv3")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_3")(x)
        x=layers.Conv2D(filters=128,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv4")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_4")(x)
        x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
        x=layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv5")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_5")(x)
        x=layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv6")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_6")(x)
        x=layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv7")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_7")(x)
        x=layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv8")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_8")(x)
        x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv9")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_9")(x)
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv10")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_10")(x)
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv11")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_11")(x)
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv12")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_12")(x)
        x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv13")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_13")(x)
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv14")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_14")(x)
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv15")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_15")(x)
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv16")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_16")(x)
        x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
        x=layers.Flatten()(x)
        x=layers.Dense(units=self.output_classes,use_bias=True,bias_initializer="zeros",kernel_initializer=self.initializer,activation=None,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_dense2")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("softmax",name="output_18")(x)
        model=models.Model(inputs,x,name="VGG-19")
        tensors=[]
        for layer in range(len(model.layers)):
            if "tensor" in model.layers[layer].name:
                tensors.append(layer)
        return model,tensors

class Resnet18():
    def __init__(self,shape,batchnorm=False,decay=1e-4,output_classes=200,activation=True,**kwargs):
        self.output_classes=output_classes
        self.shape=shape
        self.decay=decay
        self.activation=activation
        self.batchnorm=batchnorm
        self.initializer=tf.keras.initializers.VarianceScaling(scale=2.0,mode="fan_avg",distribution="normal")
    def build(self):
        inputs=layers.Input(shape=self.shape,name="output_0")
        x=layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv1")(inputs)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_1")(x)
        shortcut=x # layer 1: block 1
        stride=1
        x=layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv2")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_2")(x)        
        x=layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv3")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if x.shape[1:]!=shortcut.shape[1:] or stride!=1:
            shortcut=layers.Conv2D(filters=64,kernel_size=(1,1),padding="valid",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_shortcut_conv1")(shortcut)
        if self.batchnorm:
            shortcut=layers.BatchNormalization()(inputs=shortcut,training=True)
        x=x+shortcut
        if self.activation:
            x=layers.Activation("relu",name="output_3")(x)
        shortcut=x # layer 2: block 2
        stride=1
        x=layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv4")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_4")(x)        
        x=layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv5")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if x.shape[1:]!=shortcut.shape[1:] or stride!=1:
            shortcut=layers.Conv2D(filters=64,kernel_size=(1,1),padding="valid",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_shortcut_conv2")(shortcut)
        if self.batchnorm:
            shortcut=layers.BatchNormalization()(inputs=shortcut,training=True)
        x=x+shortcut
        if self.activation:
            x=layers.Activation("relu",name="output_5")(x)
        shortcut=x # layer 2: block 1
        stride=2
        x=layers.Conv2D(filters=128,kernel_size=(3,3),padding="same",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv6")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_6")(x)        
        x=layers.Conv2D(filters=128,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv7")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if x.shape[1:]!=shortcut.shape[1:] or stride!=1:
            shortcut=layers.Conv2D(filters=128,kernel_size=(1,1),padding="valid",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_shortcut_conv3")(shortcut)
        if self.batchnorm:
            shortcut=layers.BatchNormalization()(inputs=shortcut,training=True)
        x=x+shortcut
        if self.activation:
            x=layers.Activation("relu",name="output_7")(x)
        shortcut=x # layer 2: block 2:
        stride=1
        x=layers.Conv2D(filters=128,kernel_size=(3,3),padding="same",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv8")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_8")(x)        
        x=layers.Conv2D(filters=128,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv9")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if x.shape[1:]!=shortcut.shape[1:] or stride!=1:
            shortcut=layers.Conv2D(filters=128,kernel_size=(1,1),padding="valid",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_shortcut_conv4")(shortcut)
        if self.batchnorm:
            shortcut=layers.BatchNormalization()(inputs=shortcut,training=True)
        x=x+shortcut
        if self.activation:
            x=layers.Activation("relu",name="output_9")(x)
        shortcut=x # layer 3: block 1:
        stride=2
        x=layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv10")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_10")(x)        
        x=layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv11")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if x.shape[1:]!=shortcut.shape[1:] or stride!=1:
            shortcut=layers.Conv2D(filters=256,kernel_size=(1,1),padding="valid",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_shortcut_conv5")(shortcut)
        if self.batchnorm:
            shortcut=layers.BatchNormalization()(inputs=shortcut,training=True)
        x=x+shortcut
        if self.activation:
            x=layers.Activation("relu",name="output_11")(x)
        shortcut=x # layer 3: block 2:
        stride=1
        x=layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv12")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_12")(x)        
        x=layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv13")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if x.shape[1:]!=shortcut.shape[1:] or stride!=1:
            shortcut=layers.Conv2D(filters=256,kernel_size=(1,1),padding="valid",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_shortcut_conv6")(shortcut)
        if self.batchnorm:
            shortcut=layers.BatchNormalization()(inputs=shortcut,training=True)
        x=x+shortcut
        if self.activation:
            x=layers.Activation("relu",name="output_13")(x)
        shortcut=x # layer 4: block 1:
        stride=2
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv14")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_14")(x)        
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv15")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if x.shape[1:]!=shortcut.shape[1:] or stride!=1:
            shortcut=layers.Conv2D(filters=512,kernel_size=(1,1),padding="valid",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_shortcut_conv7")(shortcut)
        if self.batchnorm:
            shortcut=layers.BatchNormalization()(inputs=shortcut,training=True)
        x=x+shortcut
        if self.activation:
            x=layers.Activation("relu",name="output_15")(x)
        shortcut=x # layer 4: block 2:
        stride=1
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv16")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("relu",name="output_16")(x)        
        x=layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_conv17")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if x.shape[1:]!=shortcut.shape[1:] or stride!=1:
            shortcut=layers.Conv2D(filters=512,kernel_size=(1,1),padding="valid",strides=(stride,stride),activation=None,use_bias=False,kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_shortcut_conv8")(shortcut)
        if self.batchnorm:
            shortcut=layers.BatchNormalization()(inputs=shortcut,training=True)
        x=x+shortcut
        if self.activation:
            x=layers.Activation("relu",name="output_17")(x)
        x=layers.AveragePooling2D(pool_size=(8,8),padding="valid")(x)
        x=layers.Flatten()(x)
        x=layers.Dense(units=self.output_classes,use_bias=False,kernel_initializer=self.initializer,activation=None,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name="tensor_dense1")(x)
        if self.batchnorm:
            x=layers.BatchNormalization()(inputs=x,training=True)
        if self.activation:
            x=layers.Activation("softmax",name="output_18")(x)
        model=models.Model(inputs,x,name="ResNet-18")
        tensors=[]
        for layer in range(len(model.layers)):
            if "tensor" in model.layers[layer].name:
                tensors.append(layer)
        return model,tensors
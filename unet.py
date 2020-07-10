import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input,Conv1D, Conv2D, Concatenate, MaxPool2D, Conv2DTranspose
import numpy as np



def double_conv(out_filters):
    m = Sequential()
    m.add(Conv2D(filters = out_filters, kernel_size=(3,3), activation='relu'))
    m.add(Conv2D(filters = out_filters, kernel_size=(3,3), activation='relu'))
    return m



def crop_tensor(t,target_size):
    """Central crop image
    """
    tensor_size = t.get_shape()[1]
    print(tensor_size)
    adjust = (tensor_size - target_size)//2    
    t_cropped = t[:,adjust:tensor_size-adjust,adjust:tensor_size-adjust,:]
    return t_cropped

def Unet():
    input = Input(shape=(572,572,1))

    #Contraction Block
    x1 = double_conv(64)(input)
    x2 = MaxPool2D(pool_size=(2,2), strides=2)(x1)

    x3 = double_conv(128)(x2)
    x4 = MaxPool2D(pool_size=(2,2), strides=2)(x3)

    x5 = double_conv(256)(x4)
    x6 = MaxPool2D(pool_size=(2,2), strides=2)(x5)

    x7 = double_conv(512)(x6)
    x8 = MaxPool2D(pool_size=(2,2), strides=2)(x7)

    x9= Conv2D(filters = 1024, kernel_size=(3,3), activation='relu')(x8) #No MaxPool after this

    x10= Conv2D(filters = 1024, kernel_size=(3,3), activation='relu')(x9)



    #Expansion Block

    y1 = Conv2DTranspose(filters = 512,kernel_size=(2,2) ,strides = 2)(x10)
    x7_cropped = crop_tensor(x7,56)
    y2 = Concatenate()([x7_cropped,y1])
    y3 = double_conv(512)(y2)


    y4 =  Conv2DTranspose(filters = 256,kernel_size=(2,2) ,strides = 2)(y3)
    x5_cropped = crop_tensor(x5,104)
    y5 = Concatenate()([x5_cropped,y4])
    y6 = double_conv(256)(y5)

    y7 =  Conv2DTranspose(filters = 128,kernel_size=(2,2) ,strides = 2)(y6)
    x3_cropped = crop_tensor(x3,200)
    y8 =  Concatenate()([x3_cropped,y7])
    y9 = double_conv(128)(y8)
    
    y10 =  Conv2DTranspose(filters = 64,kernel_size=(2,2) ,strides = 2)(y9)
    x1_cropped = crop_tensor(x1,392)
    y11 = Concatenate()([x1_cropped,y10])
    y12 = double_conv(64)(y11)


    output = Conv2D(filters = 2, kernel_size=(1,1))(y12)

  
    model = Model(input,output)

    return model

unet_model = Unet()

unet_model.compile(tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics =["accuracy"])


input_ary = np.random.random((2,572,572,1))

pred = unet_model(input_ary)

print(pred.shape)
pred

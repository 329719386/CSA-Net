# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 03:14:09 2019

@author: Dell
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as ply
from keras import layers
from keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import Input
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from rsh.ranNewASPP_New import New_ASPP_New
from tensorflow.python.keras.backend import flatten
from tensorflow.python.keras.backend import binary_crossentropy
from tensorflow.python.keras.backend import sum
import math
from tensorflow import expand_dims
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import BatchNormalization, UpSampling2D, MaxPooling2D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D,Lambda
from tensorflow.python.keras.layers import Conv2D, Activation, \
    Concatenate, Add
from keras.backend import expand_dims

alpha = 1
tf.compat.v1.disable_eager_execution()
print(tf.__version__)

def oneD(input, _kenrel_size, return_filter_num, stride, _dilation_rate_list, _name):
    x = Conv2D(return_filter_num, _kenrel_size, padding='same', activation=None, strides=stride,
               dilation_rate=_dilation_rate_list[0], name=_name + '_1')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(return_filter_num, _kenrel_size, padding='same', activation=None, strides=stride,
               dilation_rate=_dilation_rate_list[1], name=_name + '_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(return_filter_num, _kenrel_size, padding='same', activation=None, strides=stride,
               dilation_rate=_dilation_rate_list[2], name=_name + '_3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def New_ASPP_New(input,filter):
    reduceBlock = Conv2D(filters=256, kernel_size=(1, 1), strides=1, activation=None, padding='valid',
                         kernel_initializer='he_normal',
                         name="reduceFeature")(input)
    ###################################全局平均池化###################################
    pooling_block = GlobalAveragePooling2D(name="GlobalAveragePooling2D")(reduceBlock)
    pooling_block = expand_dims(expand_dims(pooling_block, 1), 1)
    pooling_block = Conv2D(filters=256, kernel_size=(1, 1), strides=1, activation="relu", padding='valid',
                           kernel_initializer='he_normal',
                           name="pooling_block_conv1x1")(pooling_block)
    pooling_block = BatchNormalization()(pooling_block)
    pooling_block = UpSampling2D(size=(32, 32), name="upSample_block", interpolation='bilinear')(pooling_block)
    #################################################################################

    atrous_block123 = oneD(reduceBlock, (3, 3), 256, 1, [1, 2, 3], "atrous_123")#k=3,5,7  R=3,7,13
    atrous_block135 = oneD(reduceBlock, (3, 3), 256, 1, [1, 3, 5], "atrous_135")#k=3,7,11 R=3,9,19
    atrous_block139 = oneD(reduceBlock, (3, 3), 256, 1, [1, 3, 9], "atrous_139")#k=3,7,19 R=3,9,27

    total_layers = Add()([reduceBlock,pooling_block,atrous_block123,atrous_block135,atrous_block139])
    #total_layers = Concatenate()([reduceBlock, pooling_block, atrous_block123, atrous_block135, atrous_block139])
    result_ASPP = Conv2D(filters=filter, kernel_size=(1, 1), strides=1, activation=None, padding='valid',
                         kernel_initializer='he_normal',
                         name="result_ASPP")(total_layers)
    return result_ASPP

def get_flops_params():
    sess = tf.compat.v1.Session()
    graph = sess.graph
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))


def MECA(inputs1,inputs2):
    # 注意力机制
    inputs_channels1 = inputs1.shape[-1]
    c1 = int(inputs_channels1)
    gamma = 2
    b = 1
    k1 = int(abs((math.log(c1, 2) + b) / gamma))

    inputs_channels2 = inputs2.shape[-1]
    c2 = int(inputs_channels2)
    gamma = 2
    b = 1
    k2 = int(abs((math.log(c2, 2)+b)/gamma))

    x = tf.keras.layers.GlobalAveragePooling2D()(inputs1)
    x = expand_dims(x, 1)
    x = tf.keras.layers.Conv1D(1, kernel_size=k1, padding="same")(x)

    x_max = tf.keras.layers.GlobalAveragePooling2D()(inputs2)
    x_max = expand_dims(x_max, 1)
    x_max = tf.keras.layers.Conv1D(1, kernel_size=k2, padding="same")(x_max)

    x = tf.add(x, x_max)
    x = Dense(units=inputs_channels2)(x)
    x = expand_dims(x, 1)
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.keras.layers.multiply([inputs2, x])
    return x

def F(inputs1, lays):

    inputs2 = New_ASPP_New(inputs1, lays)
    inputs_channels1 = inputs1.shape[-1]
    c1 = int(inputs_channels1)
    gamma = 2
    b = 1
    k1 = int(abs((math.log(c1, 2) + b) / gamma))

    inputs_channels2 = inputs2.shape[-1]
    c2 = int(inputs_channels2)
    gamma = 2
    b = 1
    k2 = int(abs((math.log(c2, 2) + b) / gamma))

    x = tf.keras.layers.GlobalAveragePooling2D()(inputs1)
    #input1 =(1,32,32,512);x=

    x = expand_dims(x, 1)

    x = tf.keras.layers.Conv1D(1, kernel_size=k1, padding="same")(x)

    x_max = tf.keras.layers.GlobalAveragePooling2D()(inputs2)

    x_max = expand_dims(x_max, 1)

    x_max = tf.keras.layers.Conv1D(1, kernel_size=k2, padding="same")(x_max)

    x = tf.add(x, x_max)
    #(?,1,1)
    x = Dense(units=inputs_channels2)(x)
    # (?,1,1024)
    x = expand_dims(x, 1)
    # (?,1,1,1024)
    x = tf.keras.layers.Activation('sigmoid')(x)
    # (?,1,1,1024)
    x = tf.keras.layers.multiply([inputs2, x])
    # (?,32,32,1024)
    return x

def F2(inputs):
    inputs_channels1 = inputs.shape[-1]
    c1 = int(inputs_channels1)
    gamma = 2
    b = 1
    k1 = int(abs((math.log(c1, 2) + b) / gamma))

    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = expand_dims(x, 1)
    x = tf.keras.layers.Conv1D(1, kernel_size=k1, padding="same")(x)

    x = Dense(units=inputs_channels1)(x)
    x = expand_dims(x, 1)
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.keras.layers.multiply([inputs, x])
    return x

def CSA_Net (input_size=(256, 256, 3)):
    inputs = Input(input_size)
    # 特征提取
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='stage1')(conv1)
    conv1_1 = MECA(inputs, conv1)
    #(256,256,64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='stage2')(conv2)
    conv2_1 = MECA(pool1, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='stage3')(conv3)
    conv3_1 = MECA(pool2, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='stage4')(conv4)
    # conv4_1 = MECA(pool3, conv4)
    #drop4 = Dropout(0.2)(conv4)

    conv5 = F(conv4, 1024)
    #(32,32,1024)

    # 上采样
    #drop5 = Dropout(0.2)(conv5)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',name = 'conv6')(conv5)
    #(32,32,512)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='stage5')(conv6)

    pred1 = F2(conv6)
    pred1 = Conv2D(1, 1, activation='sigmoid', name="pred1")(pred1)# 预测结果1
    lossLay1 = UpSampling2D(size=(8, 8), name="lossLay1", interpolation='bilinear')(pred1)  # 计算损失函数用
    pred1 = UpSampling2D(interpolation='bilinear')(pred1)

    up7 = Conv2D(256, 1, activation='relu', strides=1, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6))
    merge7 = concatenate([conv3_1, up7, pred1])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='stage6')(conv7)

    pred2 = F2(conv7)
    pred2 = Conv2D(1, 1, activation='sigmoid', name="pred2")(pred2)  # 预测结果2
    lossLay2 = UpSampling2D(size=(4, 4), name="lossLay2", interpolation='bilinear')(pred2)  # 计算损失函数用
    pred2 = UpSampling2D(interpolation='bilinear')(pred2)

    up8 = Conv2D(128, 1, activation='relu', strides=1, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), interpolation='bilinear')(conv7))
    merge8 = concatenate([conv2_1, up8, pred2])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='stage7')(conv8)

    pred3 = F2(conv8)
    pred3 = Conv2D(1, 1, activation='sigmoid', name="pred3")(pred3)  # 预测结果3
    lossLay3 = UpSampling2D(size=(2, 2), name="lossLay3", interpolation='bilinear')(pred3)  # 计算损失函数用
    pred3 = UpSampling2D(interpolation='bilinear')(pred3)

    up9 = Conv2D(64, 1, activation='relu', strides=1, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), interpolation='bilinear')(conv8))
    merge9 = concatenate([conv1_1, up9, pred3])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='stage8')(conv9)

    result = Conv2D(1, 1, activation='sigmoid', name="result")(conv9)  # 预测结果4
    layers = [lossLay1, lossLay2, lossLay3]
    model = Model(inputs=inputs, outputs=result)
    myloss = MySelf_loss(layers)
    model.compile(optimizer=optimizers.Adam(lr=1e-4), loss=myloss, metrics=['accuracy'])
    model.summary()
    get_flops_params()
    return model


#############自定义损失函数#####################


def dice_loss(y_true_dice, y_pre_dice):
    smooth = 1.  # 1e-5
    y_true_f = flatten(y_true_dice)  # 将 y_true 拉伸为一维.
    y_pred_f = flatten(y_pre_dice)
    intersection = sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (sum(y_true_f * y_true_f) + sum(y_pred_f * y_pred_f) + smooth)

    # y_true_f = tf.reshape(y_true_dice, [-1])
    # y_pred_f = tf.reshape(y_pre_dice, [-1])
    # intersection = tf.reduce_sum(y_true_f * y_pred_f)
    # score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    loss = 1 - score
    return loss


######################################################

def dice_coef(y_true, y_pred, layers):
    # ========================Tang=================================================
    lossPred0 = 0.1 * binary_crossentropy(target=y_true, output=layers[0])
    lossPred1 = 0.2 * binary_crossentropy(target=y_true, output=layers[1])
    lossPred2 = 0.3 * binary_crossentropy(target=y_true, output=layers[2])
    lossPredResult = 0.4 * binary_crossentropy(target=y_true, output=y_pred)
    sumLoss = lossPred0 + lossPred1 + lossPred2 + lossPredResult
    return sumLoss
    # ==============================================================================

    # ========================DiceLOSS==============================================
    # lossPred0 = 0.1 * dice_loss(y_true_dice=y_true, y_pre_dice=layers[0])
    # lossPred1 = 0.2 * dice_loss(y_true_dice=y_true, y_pre_dice=layers[1])
    # lossPred2 = 0.3 * dice_loss(y_true_dice=y_true, y_pre_dice=layers[2])
    # lossPredResult = 0.4 * dice_loss(y_true_dice=y_true, y_pre_dice=y_pred)
    # sumLoss = lossPred0 + lossPred1 + lossPred2 + lossPredResult
    # return sumLoss
    # ==============================================================================


def MySelf_loss(layers):
    def dice(y_true, y_pred):
        return dice_coef(y_true, y_pred, layers)
    return dice

if __name__ == '__main__':
    CSA_Net()

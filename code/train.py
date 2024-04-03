import itertools
import os
import sys
from datetime import time, datetime
from math import ceil
import cv2
import tensorflow as tf
import keras
import numpy
# import tensorflow.python.keras
import pandas as pd
import numpy as np
import tensorflow.python.keras.models
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn import model_selection
# from tensorflow.python.keras.models import model_from_json, Sequential
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.models import model_from_json, Sequential
# from torch.nn import Dropout
# from tensorflow.python.keras.applications.resnet import ResNet50
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Input, Reshape, Concatenate, GlobalAveragePooling2D, BatchNormalization, Dropout, Activation, GlobalMaxPooling2D
from keras.layers import BatchNormalization
from collections import Counter

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.models as models

def data_process_ferplus():
    data = pd.read_csv('../fer2013.csv')
    labels = pd.read_csv('../fer2013new.csv')

    orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

    n_samples = len(data)
    w = 48
    h = 48

    y = np.array(labels[orig_class_names])


    X = np.zeros((n_samples, w, h, 1))
    #
    # x_train = np.zeros((28709, w, h, 1))
    # x_test = np.zeros((3589, w, h, 1))
    # x_val = np.zeros((3589, w, h, 1))
    # y_train = np.array(labels[orig_class_names])
    # y_test = np.array(labels[orig_class_names])
    # y_val = np.array(labels[orig_class_names])
    # train_cnt=0
    # test_cnt=0
    # val_cnt=0
    for i in range(n_samples):
        X[i] = np.fromstring(data['pixels'][i], dtype=int, sep=' ').reshape((h, w, 1))

    return X, y


def data_clean(X, y):
    orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

    # Using mask to remove unknown or NF images
    y_mask = y.argmax(axis=-1)
    mask = (y_mask < orig_class_names.index('unknown')) & (y_mask < orig_class_names.index('NF'))
    X = X[mask]
    y = y[mask]

    # Convert to probabilities between 0 and 1
    y = y[:, :-2] / 10

    # Add contempt to neutral and remove it
    # y[:, 0] += y[:, 7]
    y = y[:, :8]

    # Normalize image vectors
    X = X / 255.0



    return X, y


def split_data(X, y):
    test_size = ceil(len(X) * 0.1)
    # Split Data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=test_size,
                                                                      random_state=42)
    # data = pd.read_csv('../fer2013.csv')
    # labels = pd.read_csv('../fer2013new.csv')
    #
    # classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    #
    # n_samples = len(X)
    # w = 48
    # h = 48
    #
    # x_train = np.zeros((28709, w, h, 1))
    # x_test = np.zeros((3589, w, h, 1))
    # x_val = np.zeros((3190, w, h, 1))
    # y_train = np.zeros((28709, 8))
    # y_test = np.zeros((3589, 8))
    # y_val = np.zeros((3190, 8))
    # # y_train = np.array(labels[classes])
    # # y_test = np.array(labels[classes])
    # # y_val = np.array(labels[classes])
    # train_cnt = 0
    # test_cnt = 0
    # val_cnt = 0
    # for i in range(n_samples):
    #     if labels.iloc[i]['Usage'] == 'Training':
    #         x_train[train_cnt] = np.fromstring(data['pixels'][i], dtype=int, sep=' ').reshape((h, w, 1))
    #         y_train[train_cnt] = y[i]
    #         train_cnt = train_cnt + 1
    #
    #     if labels.iloc[i]['Usage'] == 'PublicTest':
    #         x_test[test_cnt] = np.fromstring(data['pixels'][i], dtype=int, sep=' ').reshape((h, w, 1))
    #         y_test[test_cnt] = y[i]
    #         test_cnt = test_cnt + 1
    #
    #     if labels.iloc[i]['Usage'] == 'PrivateTest':
    #         x_val[val_cnt] = np.fromstring(data['pixels'][i], dtype=int, sep=' ').reshape((h, w, 1))
    #         y_val[val_cnt] = y[i]
    #         val_cnt = val_cnt + 1
    #
    # print(train_cnt)
    # print(test_cnt)
    # print(val_cnt)
    # # non_empty_rows = ~np.all(np.isnan(x_train), axis=1)
    # # x_train = x_train[non_empty_rows]
    # # non_empty_rows = ~np.all(np.isnan(x_test), axis=1)
    # # x_test = x_test[non_empty_rows]
    # # non_empty_rows = ~np.all(np.isnan(x_val), axis=1)
    # # x_val = x_val[non_empty_rows]
    # #
    # non_empty_rows = ~np.all(np.isnan(y_train), axis=1)
    # y_train = y_train[non_empty_rows]
    # non_empty_rows = ~np.all(np.isnan(y_test), axis=1)
    # y_test = y_test[non_empty_rows]
    # non_empty_rows = ~np.all(np.isnan(y_val), axis=1)
    # y_val = y_val[non_empty_rows]

    return x_train, y_train, x_val, y_val, x_test, y_test


def data_augmentation(x_train):
    image_aug = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,  # 解释为什么是true 解释为什么没有缩放
    )
    image_aug.fit(x_train)

    # x_train_subset1 = np.squeeze(x_train[:12])
    # print("xtrain_subset1", x_train_subset1.shape)
    # print("xtrain", x_train.shape)
    # x_train_subset2 = x_train[:12]
    # print("xtrain_subset2", x_train_subset2.shape)
    #
    # fig = plt.figure(figsize=(20, 2))
    # plt.set_cmap('gray')
    # # 显示原始图片
    # for i in range(0, len(x_train_subset1)):
    #     ax = fig.add_subplot(1, 12, i + 1)
    #     ax.imshow(x_train_subset1[i])
    # fig.suptitle('Subset of Original Training Images', fontsize=20)  # 总标题
    # plt.show()
    #
    # #显示增强后的图片
    # fig = plt.figure(figsize=(20, 2))
    #
    # for x_batch in image_aug.flow(x_train_subset2, batch_size=12, shuffle=False):
    #     for i in range(0, 12):
    #         ax = fig.add_subplot(1, 12, i + 1)
    #         ax.imshow(np.squeeze(x_batch[i]))
    #     fig.suptitle('Augmented Images', fontsize=20)  # 总标题
    #     plt.show()
    #     break;

    return image_aug


def plot_acc_loss(history):
    # Plot accuracy graph
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    # plt.ylim([0, 1.0])
    plt.legend(loc='upper left')
    #plt.show()
    plt.savefig('../assess/accuracy.png')

    # Plot loss graph
    plt.clf()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim([0, 3.5])
    plt.legend(loc='upper right')
    #plt.show()
    plt.savefig('../assess/loss.png')

def save_model_and_weights(model, test_acc):
    # Serialize and save model to JSON
    test_acc = int(test_acc * 10000)
    keras.models.save_model(model, '../Saved-Models/model' + str(test_acc) +'.h5')


def run():
    fer_classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

    X, y = data_process_ferplus()
    X, y = data_clean(X, y)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(X, y)
    # image_aug_run = data_augmentation(x_train)

    epochs = 100
    batch_size = 10

    print("X_train shape: " + str(x_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(x_test.shape))
    print("Y_test shape: " + str(y_test.shape))
    print("X_val shape: " + str(x_val.shape))
    print("Y_val shape: " + str(y_val.shape))
    model = ResNet50(weights=None, input_tensor=tf.keras.Input(shape=(48, 48, 1)),
                     include_top=False, pooling='max')

    model_git = Sequential([model,
                        Flatten(),
                        BatchNormalization(),
                        Dense(4096, activation='relu'),
                        Dropout(0.5),
                        BatchNormalization(),
                        Dense(2048, activation='relu'),
                        Dropout(0.5),
                        BatchNormalization(),
                        Dense(512, activation='relu'),
                        Dropout(0.5),
                        BatchNormalization(),
                        Dense(8, activation='softmax', name='classifer')])
    model_git.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title=None, cmap=plt.cm.Blues,save_path=None):
    #     """
    #     此函数打印并绘制混淆矩阵。
    #     可以通过设置 `normalize=True` 来将混淆矩阵中的值转化为百分比形式。
    #     """
    #     if not title:
    #         if normalize:
    #             title = 'Normalized confusion matrix'
    #         else:
    #             title = 'Confusion matrix, without normalization'
    #
    #     # 计算混淆矩阵
    #     cm = confusion_matrix(y_true, y_pred)
    #
    #     if normalize:
    #         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #
    #     plt.figure(figsize=(20, 20))
    #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #     plt.subplots_adjust(left=0.2)
    #     plt.title(title)
    #     plt.colorbar()
    #     tick_marks = np.arange(len(classes))
    #     plt.xticks(tick_marks, classes, rotation=45)
    #     plt.yticks(tick_marks, classes)
    #
    #     fmt = '.2f' if normalize else 'd'
    #     thresh = cm.max() / 2.
    #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #         plt.text(j, i, format(cm[i, j], fmt),
    #                  horizontalalignment="center",
    #                  color="white" if cm[i, j] > thresh else "black")
    #
    #     plt.tight_layout()
    #     plt.ylabel('True label')
    #     plt.xlabel('Predicted label')
    #     # plt.show()
    #     plt.savefig(save_path)
    #
    # # 定义混淆矩阵回调函数
    # class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    #     def __init__(self, x_val, y_val, class_names):
    #         super(ConfusionMatrixCallback, self).__init__()
    #         self.x_val = x_val
    #         self.y_val = y_val
    #         self.class_names = class_names
    #
    #     def on_epoch_end(self, epoch, logs=None):
    #         y_pred = np.argmax(self.model.predict(self.x_val), axis=1)
    #         y_true = np.argmax(self.y_val, axis=1)
    #         save_path = f"../feature_maps/confusion_matrix_epoch_{epoch+1}.png"
    #         plot_confusion_matrix(y_true, y_pred, classes=self.class_names, title=f'Confusion matrix (Epoch {epoch+1})', save_path=save_path)

    #sparse_categorical_crossentropy
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # model_git.summary()
    # callback = ConfusionMatrixCallback(x_val, y_val, fer_classes)

    history = model_git.fit(x_train, y_train, epochs=epochs, validation_split=0.2)
    # , callbacks=[callback]

    test_loss, test_acc = model_git.evaluate(x_test, y_test, batch_size=batch_size)

    plot_acc_loss(history)
    save_model_and_weights(model_git, test_acc)



run()
# train_model()
# load_model_and_weights('../Saved-Models/model.json','../Saved-Models/model4000.h5')

#model_path = '../Saved-Models/model4000.h5'

# 加载模型
# restored_model= keras.models.load_model('../my_model.keras')



#
# # 使用Counter进行计数
# usage_counts = Counter(usages)

# 输出结果
# print("训练集样本数量:", usage_counts['Training'])
# print("私人测试集样本数量:", usage_counts['PrivateTest'])
# print("公共测试集样本数量:", usage_counts['PublicTest'])
# usages = pd.read_csv('../fer2013new.csv')
# print(usages.iloc[0]['Usage'])


# usages = pd.read_csv('../fer2013new.csv')['Usage'].values
#
# # 使用Counter进行计数
# usage_counts = Counter(usages)
#
# # 输出结果
# print("训练集样本数量:", usage_counts['Training'])
# print("私人测试集样本数量:", usage_counts['PrivateTest'])
# print("公共测试集样本数量:", usage_counts['PublicTest'])

# 训练集样本数量: 28709
# 私人测试集样本数量: 3589
# 公共测试集样本数量: 3589
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 04:50:38 2020

@author: user
"""

#引入各項需要的lib
import os
import sys
import cv2
import time
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt

#處理後的照片大小(正方形)，目的要截出眼球
imgprocess = 100
img_row, img_col = imgprocess, imgprocess

#資料處理比例分配，train data與test data，此表test data佔all data的比例
splitnum = 0.5


def data_x_y_preprocess(datapath):
    datapath = datapath
    #初始化data_train_X, data_test_X
    data_train_X = np.zeros((img_row, img_col)).reshape(1, img_row, img_col)
    data_test_X = np.zeros((img_row, img_col)).reshape(1, img_row, img_col)
    #初始化data_train_Y, data_test_Y
    data_train_Y = []
    data_test_Y = []
    #紀錄照片數量與左眼數量，確保資料平衡
    trainleft, testleft = 0, 0
    trainpictureCount, testpictureCount = 0, 0
    #宣告參數
    img_prerow, img_precol = 640, 480
    blur1 = 5
    num_class = 2
    for root, dirs, files in os.walk(datapath):
        if len(files) != 0:
            print('\n' + str(len(files)))
            random.shuffle(files)  #打亂files的順序，使每次訓練資料都是打散的
        for f in files:
            label = int(root.split("\\")[1])  #取圖片的label
            fullpath = os.path.join(root, f)  #求出圖片的path
            img = cv2.imread(fullpath)  #讀取圖片

            #前處理，壓縮，轉灰階，標準化，去除雜訊
            img = cv2.resize(img, (img_prerow // 2, img_precol // 2))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_64F)
            img = stdlize(img)

            tmp = img

            #前處理，處理邊緣，並消除雜訊，目的為清理出容易辨識之圖片
            img = cv2.Laplacian(img, cv2.CV_64F)
            img = cv2.GaussianBlur(img, (blur1, blur1), 0)

            #前處理，處理邊緣(不同方法)，標準化後，以cv2.HOUGH_GRADIENT曾測圓形物，藉此找出眼球
            tmp = cv2.Sobel(tmp, cv2.CV_64F, 1, 0, ksize=5)
            tmp = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_64F)
            tmp = cv2.convertScaleAbs(tmp)
            circle = cv2.HoughCircles(tmp,
                                      cv2.HOUGH_GRADIENT,
                                      2,
                                      25,
                                      param1=100,
                                      param2=20,
                                      minRadius=50,
                                      maxRadius=175)

            #若沒找到圓，則捨棄照片，並記錄
            if type(circle) != np.ndarray:
                print("########################################")
                continue

            circles = circle[0, :, :]  # 提取為二維
            circles = np.uint16(np.around(circles))  # 四捨五入，取整

            tmpnum = (img_prerow // 4) * (img_prerow // 4) + (
                img_precol // 4) * (img_precol // 4)
            xidx, yidx = 0, 0
            for i in circles[:]:  #取最靠近中心圓
                num = (i[0] -
                       (img_prerow // 4)) * (i[0] - (img_prerow // 4)) + (
                           i[1] - (img_precol // 4)) * (i[1] -
                                                        (img_precol // 4))
                if tmpnum > num:
                    tmpnum = num
                    xidx = i[0]
                    yidx = i[1]
                #確保不超出圖片範圍
                if xidx - imgprocess < 0:
                    xidx = imgprocess
                elif xidx + imgprocess > (img_prerow // 2):
                    xidx = (img_prerow // 2) - imgprocess
                if yidx - imgprocess < 0:
                    yidx = imgprocess
                elif yidx + imgprocess > (img_precol // 2):
                    yidx = (img_precol // 2) - imgprocess

            #將圖片切割，目的為找出眼球，並且縮小圖片(4倍)
            img = img[yidx - imgprocess:yidx + imgprocess:2,
                      xidx - imgprocess:xidx + imgprocess:2]
            img = (np.array(img) / 255).reshape(1, img_row, img_col)  #標準化圖片

            if (trainpictureCount +
                    testpictureCount) * splitnum > testpictureCount:
                #應該將這筆資料放入test中
                if f.find('Left') != -1:
                    if testleft > trainleft:
                        #若test中Left過多，則放入train
                        data_train_Y.append(label)  #label存入data_y
                        data_train_X = np.vstack(
                            (data_train_X, img))  #將圖片放入data_x
                        trainpictureCount += 1
                        trainleft += 1
                    else:
                        #反之，則放入test
                        data_test_Y.append(label)  #label存入data_y
                        data_test_X = np.vstack(
                            (data_test_X, img))  #將圖片放入data_x
                        testpictureCount += 1
                        testleft += 1
                elif f.find('Right') != -1:
                    if testpictureCount - testleft > trainpictureCount - trainleft:
                        #若test中Right過多，則放入train
                        data_train_Y.append(label)  #label存入data_y
                        data_train_X = np.vstack(
                            (data_train_X, img))  #將圖片放入data_x
                        trainpictureCount += 1
                    else:
                        #反之，則放入test
                        data_test_Y.append(label)  #label存入data_y
                        data_test_X = np.vstack(
                            (data_test_X, img))  #將圖片放入data_x
                        testpictureCount += 1
            else:
                #應該將這筆資料放入train中
                if f.find('Left') != -1:
                    if testleft < trainleft:
                        #若train中Left過多，則放入test
                        data_test_Y.append(label)  #label存入data_y
                        data_test_X = np.vstack(
                            (data_test_X, img))  #將圖片放入data_x
                        testpictureCount += 1
                        testleft += 1
                    else:
                        #反之，則放入train
                        data_train_Y.append(label)  #label存入data_y
                        data_train_X = np.vstack(
                            (data_train_X, img))  #將圖片放入data_x
                        trainpictureCount += 1
                        trainleft += 1
                elif f.find('Right') != -1:
                    if testpictureCount - testleft < trainpictureCount - trainleft:
                        #若train中Right過多，則放入test
                        data_test_Y.append(label)  #label存入data_y
                        data_test_X = np.vstack(
                            (data_test_X, img))  #將圖片放入data_x
                        testpictureCount += 1
                    else:
                        #反之，則放入train
                        data_train_Y.append(label)  #label存入data_y
                        data_train_X = np.vstack(
                            (data_train_X, img))  #將圖片放入data_x
                        trainpictureCount += 1

            #顯示進度，路徑，分配圖片數量、左右眼數量確認
            tmps = datapath + ' ' + str(trainpictureCount) + ' ' + str(
                trainleft) + ' ' + str(testpictureCount) + ' ' + str(testleft)
            sys.stdout.write("\r%s" % tmps)
            sys.stdout.flush()

    data_train_X = np.delete(data_train_X, [0], 0)  #刪除原先宣告的np.zeros
    data_train_X = data_train_X.reshape(trainpictureCount, img_row, img_col,
                                        1)  #調整格式

    data_test_X = np.delete(data_test_X, [0], 0)  #刪除原先宣告的np.zeros
    data_test_X = data_test_X.reshape(testpictureCount, img_row, img_col,
                                      1)  #調整格式

    #將label轉成one-hot encoding
    data_train_Y = np_utils.to_categorical(data_train_Y, num_class)
    data_test_Y = np_utils.to_categorical(data_test_Y, num_class)

    return data_train_X, data_train_Y, data_test_X, data_test_Y  #回傳所有處理好的資料


#圖片前處理並標準化
def stdlize(img):
    mean, stdev = cv2.meanStdDev(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = (img[i][j] - mean) / stdev
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_64F)
    return img


start = time.time()
model = Sequential()  #建立線性執行模型

#建立卷基層，filters=32，即output size層數，kernel_size=5*5，並且使用relu
model.add(
    Conv2D(32,
           kernel_size=(5, 5),
           activation='relu',
           input_shape=(img_row, img_col, 1)))
model.add(MaxPooling2D(pool_size=(4, 4)))  #建立池化層，大小4*4，取最大值

model.add(Dropout(0.5))  #防止過度擬合，隨機Dropout神經元，段開比例0.5

#建立卷基層，filters=64，即output size層數，kernel_size=5*5，並且使用relu
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))  #建立池化層，大小4*4，取最大值

model.add(Dropout(0.4))  #防止過度擬合，隨機Dropout神經元，段開比例0.4
model.add(Flatten())  #使用Flatten將維的輸入一維化，使用於卷積層道全連接層的過度

model.add(Dropout(0.3))  #防止過度擬合，隨機Dropout神經元，段開比例0.3
model.add(Dense(256, activation='relu'))  #建立全連階層，256個輸出，並且使用relu

model.add(Dropout(0.2))  #防止過度擬合，隨機Dropout神經元，段開比例0.2
model.add(Dense(32, activation='relu'))  #建立全連階層，64個輸出，並且使用relu

model.add(Dropout(0.1))  #防止過度擬合，隨機Dropout神經元，段開比例0.1
model.add(Dense(units=2, activation='softmax'))  #建立全連階層，2個輸出，並且使用softmax

model.summary()  #總結輸出現在連接狀態

#編譯，選擇損失函數、優化方式及成效衡量方式
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

#從train_image資料夾取得資料
datapath = "image"
load = time.time()
data_train_X, data_train_Y, data_test_X, data_test_Y = data_x_y_preprocess(
    datapath)

#訓練過程，儲存於train_history之中，代數1000，且用0.2的資料做檢查，並且每代資料打亂
startfit = time.time()
train_history = model.fit(data_train_X,
                          data_train_Y,
                          batch_size=32,
                          epochs=1000,
                          verbose=1,
                          validation_split=0.2,
                          shuffle=True)

#顯示損失函數，個階段訓練成果
startevaluate = time.time()
score = model.evaluate(data_test_X, data_test_Y, verbose=0)
end = time.time()

model.save('iris.model')

print()
#輸出各項時間統計數字
print('Create model costs :', (load - start) // 60, 'minutes',
      (load - start) % 60, 'seconds')
print('Load all data costs :', (startfit - load) // 60, 'minutes',
      (startfit - load) % 60, 'seconds')
print('Training costs :', (startevaluate - startfit) // 60, 'minutes',
      (startevaluate - startfit) % 60, 'seconds')
print('Evaluate costs :', (end - startevaluate) // 60, 'minutes',
      (end - startevaluate) % 60, 'seconds')
print('Total costs :', (end - start) // 60, 'minutes', (end - start) % 60,
      'seconds')
print()
print('Test loss:', score[0])  #輸出Test loss
print('Test accuracy:', score[1])  #輸出Test accuracy

plt.subplot(121)
plt.plot(train_history.history['accuracy'])  #輸出訓練過程中的accuracy
plt.plot(train_history.history['val_accuracy'])  #輸出訓練過程中的val_accuracy
plt.title('Train History')  #圖片title
plt.ylabel('accuracy')  #圖片ylabel
plt.xlabel('Epoch')  #圖片xlabel
plt.legend(['accuracy', 'val_accuracy'], loc='lower right')  #圖片圖例

plt.subplot(122)
plt.plot(train_history.history['loss'])  #輸出訓練過程中的loss
plt.plot(train_history.history['val_loss'])  #輸出訓練過程中的val_loss
plt.title('Train History')  #圖片title
plt.ylabel('loss')  #圖片ylabel
plt.xlabel('Epoch')  #圖片xlabel
plt.legend(['loss', 'val_loss'], loc='upper left')  #圖片圖例

plt.show()  #輸出所有訓練資料

import os #os model
import random
import keras
import numpy as np
from PIL import Image   # PIL 提供處理 image 的模組
from keras.models import Sequential # 建立最簡單的線性模組，標準一層一層傳遞的神經網路
from keras.layers import Dense, Dropout, Flatten    # 引入層數
from keras.layers import Conv2D, MaxPooling2D   # CNN 卷積層和池化層
from keras.models import load_model # 載入模型
from keras.utils import np_utils    # 用來後續將 Label 轉為 one-hot-encoding
from matplotlib import pyplot as plt    # 用來繪圖

def data_x_y_preprocess(datapath):  # data_x 和 data_y (Label) 前處理函式
    img_row, img_col = 28, 28   # 定義圖片大小
    # datapath = datapath    # 訓練資料路徑
    allList = os.walk(datapath)   # 在目錄樹中遊走輸出目錄中的文件名
    data_x = np.zeros((28, 28)).reshape(1, 28, 28) # 儲存圖片
    picCount = 0    # 紀錄圖片數量
    data_y = [] # 紀錄label
    num_class = 10    # 數字種類 10 種
    for root, dirs, files in allList:   # 讀取 img 資料夾內所有檔案
        for f in files:
            label = int(root.split("\\")[4])  # split'\\'後，取得label(第四個)
            data_y.append(label)    # 將 label 放入data_y
            fullpath = os.path.join(root, f) # 取得檔案路徑
            print(fullpath)
            img = Image.open(fullpath)  # 開啟 img
            img = (np.array(img) / 255).reshape(1, 28, 28)  # 正確讀出照片並 reshape
            data_x = np.row_stack((data_x, img))   # 行合併
            picCount += 1   # 圖片數量 + 1
    data_x = np.delete(data_x, [0], 0)  # 刪除一開始宣告的 np.zero
    data_x = data_x.reshape(picCount, img_row, img_col, 1)   # 調整資料格式
    data_y = np_utils.to_categorical(data_y, num_class)    # 將 label 轉為 one-hot-encoding
    return data_x, data_y

# 建立 CNN 模型
model = Sequential()
# 建立卷積層，filter = 32，即 outer space 深度，Kernal size = 5x5，activation function 採用 relu，padding:一律補零，data_format:為預設值
model.add(Conv2D(32, (5, 5), activation = 'relu', padding = "same", data_format = "channels_last", input_shape = (28, 28, 1)))
# 建立卷積層，filter = 32，即 outer space 深度，Kernal size = 5x5，activation function 採用 relu，padding:一律補零，data_format:為預設值
model.add(Conv2D(32, (5, 5), activation = 'relu', padding = "same", data_format = "channels_last", input_shape = (28, 28, 1)))
# 建立池化層，池化大小 = 2x2，取最大值，data_format: channels_last 是指通道(色彩)資料放在最後一維
model.add(MaxPooling2D((2, 2), data_format = "channels_last"))
# Dropout 層隨機斷開輸入神經元，用於防止過度擬合，斷開比例 = 0.5
model.add(Dropout(0.5))

# 建立卷積層，filter = 64，即 outer space 深度，Kernal size = 3x3，activation function 採用 relu，padding:一律補零，data_format:為預設值
model.add(Conv2D(64, (3, 3), activation = 'relu', padding = "same", data_format = "channels_last"))
# 建立卷積層，filter = 64，即 outer space 深度，Kernal size = 3x3，activation function 採用 relu，padding:一律補零，data_format:為預設值
model.add(Conv2D(64, (3, 3), activation = 'relu', padding = "same", data_format = "channels_last"))
# 建立池化層，池化大小 = 2x2，取最大值，data_format: channels_last 是指通道(色彩)資料放在最後一維
model.add(MaxPooling2D((2, 2), data_format = "channels_last"))
# Dropout 層隨機斷開輸入神經元，用於防止過度擬合，斷開比例 = 0.5
model.add(Dropout(0.5))

# Flatten 層把多維的輸入一維化，常用在卷積層到全連階層的過度
model.add(Flatten())
# 全連接層:256個 output
model.add(Dense(256, activation = 'relu'))
# Dropout 層隨機斷開輸入神經元，用於防止過度擬合，斷開比例 = 0.5
model.add(Dropout(0.5))
# 使用 softmax activation funciotn 將結果分類(units = 10)
model.add(Dense(units = 10, activation = 'softmax'))


# 編譯:選擇損失函數，優化方法和成效衡量方式
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
# 呼叫data_x_y_preprocess，並傳入路徑
data_train_X, data_train_Y = data_x_y_preprocess(r"C:\handwrite__detect\train_image")
# 呼叫data_x_y_preprocess，並傳入路徑
data_test_X, data_test_Y = data_x_y_preprocess(r"C:\handwrite__detect\test_image")
# 進行訓練訓練過程會存在 train_history
train_history = model.fit(data_train_X, data_train_Y, batch_size = 300, epochs = 150, verbose = 2, validation_split = 0.1)
# 顯示損失函數和訓練成果(分數)
score = model.evaluate(data_test_X, data_test_Y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 優化過程曲線
plt.plot(train_history.history['accuracy']) # 將 accuracy 放到函數內
plt.plot(train_history.history['val_accuracy']) # 將 val_accuracy 放到函數內
plt.title('Train History')  # 設定 title
plt.xlabel('Epoch') # 設定 x 軸 label
plt.ylabel('loss')  # 設定 y 軸 label
plt.legend(['accuracy', 'val_accuracy'], loc = 'upper left')  # 顯示數據名稱
plt.show()  # 顯示
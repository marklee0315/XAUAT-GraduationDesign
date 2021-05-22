import os
import random

import numpy as np
import openpyxl
import pandas as pd
from keras.layers import Activation, Flatten, Convolution1D, MaxPooling1D
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot
from numpy import array
from sklearn.preprocessing import OneHotEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---- 数据导入 ----
data = pd.read_excel("./data.xlsx")

origin_data_x = data.iloc[:, 1:].values  # 提取所有行，然后提取第2列开始的所有列
origin_data_y = data.iloc[:, 0].values  # 提取所有行，第2列

index = [j for j in range(len(origin_data_x))]
random.shuffle(index)
origin_data_y = origin_data_y[index]
origin_data_x = origin_data_x[index]

# ---- 参数定义----
split_point = int(len(origin_data_x) * 0.8)  # 测试集与训练集2、8分
input_size = 140
time_step = 1
labels = 10
epochs = 500
batch_size = 100


# 标准化，工具函数
def normal(datas):
    hang = datas.shape[0]

    list1 = []
    for i in range(hang):
        n = 0
        meihang = datas[i]

        list2 = []
        for j in range(5):
            zhen = meihang[n:n + 28]

            jishu = zhen[::2]  # 取奇数位，即x
            x_max = max(jishu)
            x_min = min(jishu)
            x_err = x_max - x_min
            if x_err == 0:
                x_err = 0.001

            oushu = zhen[1::2]  # 取偶数位，即y
            y_max = max(oushu)
            y_min = min(oushu)
            y_err = y_max - y_min
            if y_err == 0:
                y_err = 0.001
            for k in range(14):
                x_n = (jishu[k] - x_min) / x_err
                y_n = (oushu[k] - y_min) / y_err
                x_n = str(x_n)
                y_n = str(y_n)

                list2.append(x_n)
                list2.append(y_n)

            n += 1
        list1.append(list2)

    normal_data = np.array(list1)

    return normal_data


# 对labels进行one-hot编码
def label2hot(labels):
    values = array(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = values.reshape(len(values), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded


hot_data_y = label2hot(origin_data_y[:])

# 训练集数据
train_x = origin_data_x[:split_point]
train_x = normal(train_x)
train_x = train_x.reshape([-1, input_size, time_step])
train_y = hot_data_y[:split_point]
print(train_x.shape)
print(train_x.shape[1])
print(train_x.shape[2])

# 测试集数据
test_x = origin_data_x[split_point:]
test_x = normal(test_x)
test_x = test_x.reshape([-1, input_size, time_step])
test_y = hot_data_y[split_point:]

print("数据处理完成!")

# 模型结构

model = Sequential()
model.add(Convolution1D(filters=128, kernel_size=28, strides=1, input_shape=(140, 1)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=None))
model.add(Convolution1D(filters=64, kernel_size=28))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=None))
model.add(Flatten())
model.add(Dropout(0.05))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(labels, activation='softmax'))
model.summary()

# 自定义优化器参数
sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 训练
history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_y), verbose=2,
                    shuffle=True)
# 保存模型文件
model.save('./models/cnn_model_label.h5')
# 评估
score = model.evaluate(test_x, test_y, verbose=2)  # evaluate函数按batch计算在某些输入数据上模型的误差
print('Test accuracy:', score[1])
score = model.evaluate(train_x, train_y, verbose=2)  # evaluate函数按batch计算在某些输入数据上模型的误差
print('Train accuracy:', score[1])

# 导出数据
prediction_label = model.predict_classes(test_x)
prediction_label = [i + 1 for i in prediction_label]
fact_label = np.argmax(test_y, 1)
fact_label = [i + 1 for i in fact_label]
analysis = [fact_label, prediction_label]
wb = openpyxl.Workbook()
sheet = wb.active
sheet.title = 'test_data'
for i in range(0, 2):
    for j in range(0, len(analysis[i])):
        sheet.cell(row=j + 1, column=i + 1, value=analysis[i][j])
wb.save('./datas/test_data.xlsx')
print("写入预测数据成功！")

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.xlabel('Epochs', fontsize=12)
pyplot.ylabel('Loss', fontsize=12)
pyplot.savefig("./images/Loss_label.png")
pyplot.show()
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.xlabel('Epochs', fontsize=12)
pyplot.ylabel('Accuracy', fontsize=12)
pyplot.savefig("./images/Accuracy_label.png")
pyplot.show()

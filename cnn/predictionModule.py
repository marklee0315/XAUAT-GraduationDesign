import os

import numpy as np
import pandas as pd
from keras.models import load_model
from numpy import array
from sklearn.preprocessing import OneHotEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---- 数据导入 ----
data = pd.read_excel('H:\Code\PyCharm\GraduationDesign\cnn\\test.xlsx', header=None)
origin_data_x = data.iloc[:, 1:].values
origin_data_y = data.iloc[:, 0].values  # 提取所有行，第1列，即label列

print(type(origin_data_x))
print(origin_data_x)


# ---- 参数定义----
input_size = 140  # 长度
time_step = 1  # 步长
labels = 10  # 分类数，共10类


# 对labels进行one-hot编码
def label2hot(labels):
    values = array(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = values.reshape(len(values), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded


hot_data_y = label2hot(origin_data_y[:])


# 标准化，工具函数
def normal(datas):
    hang = datas.shape[0]
    list1 = []
    for i in range(hang):
        n = 0
        meihang = datas[i]
        # print(meihang)
        list2 = []
        for j in range(5):
            zhen = meihang[n:n + 28]
            # print(zhen)
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
            n += 28
        list1.append(list2)
    normal_data = np.array(list1)
    return normal_data


# 测试数据
test_x = origin_data_x
test_x = normal(test_x)
test_x = test_x.reshape([-1, input_size, time_step])

# 调用训练好的模型
model = load_model('H:\Code\PyCharm\GraduationDesign\cnn\models\cnn_model_label.h5')  # 选取自己的.h模型名称

# 预测
predict = model.predict_classes(test_x)
print(predict)


def execute():
    return predict

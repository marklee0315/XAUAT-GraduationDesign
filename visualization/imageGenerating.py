import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel('H:\Code\PyCharm\Keras\old\dataset.xlsx', header=None)

x = [1.1]
y = [2.1]
x.clear()
y.clear()

shutil.rmtree('H:\Code\PyCharm\Keras\windows\predict\images\\')
os.mkdir('H:\Code\PyCharm\Keras\windows\predict\images\\')

path = 'H:\Code\PyCharm\Keras\windows\predict\images\\'
suffix = '.jpg'

for i in range(int(len(data) / 20)):
    for j in range(20):
        x.append(data[0][i * 20 + j])
        y.append(data[1][i * 20 + j])
    plt.scatter(x, y)
    # plt.title(str(actionSequence[i])) #加标题用
    plt.savefig(path + str(i) + suffix)
    plt.show()
    x.clear()
    y.clear()


def executeWith(actionSequence):
    print(actionSequence)
    print('imageGenerating ok!')

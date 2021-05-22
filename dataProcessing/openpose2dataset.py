import os

import pandas as pd


def execute():
    openposeDataPath = 'H:\Code\PyCharm\Keras\\fakedata'

    openposeDatas = os.listdir(openposeDataPath)

    for i in range(len(openposeDatas)):
        openposeFileList = [os.path.join(openposeDataPath, openposeData) for openposeData in openposeDatas]
        print(openposeFileList)
        data = pd.read_excel(openposeFileList[i], header=None)
        print(data[2][3])

    print('openpose2dataset ok!')
    print('-----------------------------------------------------')
# openposeDataPath = 'H:\Code\PyCharm\Keras\\fakedata'
#
# openposeDatas = os.listdir(openposeDataPath)
#
# for i in range(len(openposeDatas)):
#     openposeFileList = [os.path.join(openposeDataPath, openposeData) for openposeData in openposeDatas]
#     print(openposeFileList)
#     data = pd.read_excel(openposeFileList[i], header=None)
#     print(data[2][3])

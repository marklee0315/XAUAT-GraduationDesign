import tkinter.messagebox
from tkinter import *
from tkinter import filedialog

from cnn import predictionModule
from dataProcessing import openpose, openpose2dataset

root = Tk()
root.title('人体关节点行为识别')
root.geometry('800x500')


def processing_video():
    # 导入视频 数据处理

    videoPath = filedialog.askopenfilename()
    openpose.executeFrom(videoPath)
    openpose2dataset.execute()
    # actionSequence = predictionModule.execute()
    # imageGenerating.executeWith(actionSequence)
    # img2video.execute()
    tkinter.messagebox.showinfo(title='processing_video', message='成功打开视频！！')  # 提示信息对话窗
    # videoDisplay.executeFrom(videoPath)


def predict():
    tkinter.messagebox.showinfo(title='predict', message='预测开始！！')  # 提示信息对话窗
    # videoDisplay.executeFrom()

    model = predictionModule.load_model('H:\Code\PyCharm\GraduationDesign\cnn\models\cnn_model_label.h5')
    prediction = model.predict_classes(predictionModule.test_x)
    print(prediction[1])


btn1 = Button(root, text='打开视频', command=processing_video)
btn1.place(relx=0.1, rely=0.15, relwidth=0.3, relheight=0.1)

btn2 = Button(root, text='开始判断', command=predict)
btn2.place(relx=0.6, rely=0.15, relwidth=0.3, relheight=0.1)

root.mainloop()

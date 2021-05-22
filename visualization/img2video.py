import os

import cv2

img_root = 'H:\Code\PyCharm\Keras\windows\predict\images\\'
out_root = 'H:\Code\PyCharm\Keras\windows\predict\\video\outputVideo.avi'

# Edit each frame's appearing time!
fps = 5.0
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

videoWriter = cv2.VideoWriter(out_root, fourcc, fps, (640, 480))

im_names = os.listdir(img_root)
print(len(im_names))

for im_name in range(len(im_names)):
    string = img_root + str(im_name) + '.jpg'
    print(string)
    frame = cv2.imread(string)
    videoWriter.write(frame)

videoWriter.release()


def execute():
    print('img2video ok!')

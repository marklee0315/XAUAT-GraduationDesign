# Use the maskrcnn for the humanpose keypoints detection
mask rcnn can be used to the human keypoints detection


# Requirements

1、	Python3，Keras，TensorFlow。

 -   Python 3.4+
 -   TensorFlow 1.3+
 -   Keras 2.0.8+
 -   Jupyter Notebook
 -   Numpy, skimage, scipy, Pillow, cython, h5py
 -   opencv 2.0


2、MS COCO Requirements:

To train or test on MS COCO, you'll also need:

 -   pycocotools (installation instructions below)
 -   [MS COCO Dataset](http://cocodataset.org/#home)
 -   Download the 5K  [minival](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0)  and the 35K  [validation-minus-minival](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0)  subsets. More details in the original  [Faster R-CNN implementation](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md).
 
3、Download pre-trained COCO weights (mask_rcnn_coco_humanpose.h5) from the release page
4、(Optional) To train or test on MS COCO install  `pycocotools`  from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

-   Linux:  [https://github.com/waleedka/coco](https://github.com/waleedka/coco)
-   Windows:  [https://github.com/philferriere/cocoapi](https://github.com/philferriere/cocoapi). You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)
 

## Getting started
-   [demo_human_pose.ipynb](https://github.com/chrispolo/Keypoints-of-humanpose-with-Mask-R-CNN/blob/master/demo_human_pose.ipynb)  
Is the easiest way to start. It shows an example of using a model pre-trained on MS COCO to segment objects in your own images. It includes code to run humanposeestimation on arbitrary images.
是最简单的开始方式。 它显示了一个示例，该示例使用在MS COCO上预训练的模型来分割您自己的图像中的对象。 它包含用于在任意图像上运行人为估计的代码。
-   [train_human_pose.ipynb](https://github.com/chrispolo/Keypoints-of-humanpose-with-Mask-R-CNN/blob/master/train_human_pose.ipynb)  
shows how to train Mask R-CNN on your own coco 2017 dataset. This notebook introduces a toy dataset (Shapes) to demonstrate training on a new dataset.
显示了如何在您自己的coco 2017数据集上训练Mask R-CNN。 本notebook介绍了玩具数据集（形状），以演示对新数据集的训练。
-   ([model.py](https://github.com/chrispolo/Keypoints-of-humanpose-with-Mask-R-CNN/blob/master/model.py),  
     [utils.py](https://github.com/chrispolo/Keypoints-of-humanpose-with-Mask-R-CNN/blob/master/utils.py),  
     [config.py](https://github.com/chrispolo/Keypoints-of-humanpose-with-Mask-R-CNN/blob/master/config.py)): 
     These files contain the main Mask RCNN implementation.
    
    
- [inference_humanpose.ipynb](https://github.com/chrispolo/Keypoints-of-humanpose-with-Mask-R-CNN/blob/master/inference_humanpose.ipynb)  
This notebook goes in depth into the steps performed to detect and segment objects. It provides visualizations of every step of the pipeline.
本文档深入介绍了检测和分割对象所执行的步骤。 它提供了管道中每个步骤的可视化。
- [video_demo.py](https://github.com/chrispolo/Keypoints-of-humanpose-with-Mask-R-CNN/blob/master/video_demo.py)  
This document show how to detect the keypoints in the video, which use the opencv to detect frame by frame.
本文档介绍了如何检测视频中的关键点，这些关键点使用opencv逐帧检测。
## Training  on coco keypoints

I am  providing pre-trained weights for MS COCO 2017 kryoints to make it easier to start. You can use those weights as a starting point to train your own variation on the network. Training and evaluation code is in  `coco.py`. You can import this module in Jupyter notebook (see the provided notebooks for examples) or you can run it directly from the command line as such:
我为MS COCO 2017 kryoints提供了预先训练的砝码，以使其更容易上手。 您可以将这些权重用作在网络上训练自己的变体的起点。 培训和评估代码在`coco.py`中。 您可以在Jupyter笔记本中导入此模块（有关示例，请参阅提供的笔记本），也可以直接从命令行运行它，如下所示：
```
# Train a new model starting from pre-trained COCO weights  从预先训练的COCO重量开始训练新模型
python3 coco.py train --dataset=/path/to/coco/ --model=coco

# Train a new model starting from ImageNet weights   从ImageNet权重开始训练新模型
python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

# Continue training a model that you had trained earlier   继续训练先前训练的模型
python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.   继续训练您训练的最后一个模型。 这会发现模型目录中最后训练的权重。
python3 coco.py train --dataset=/path/to/coco/ --model=last
```
You can also run the COCO evaluation code with:

```
# Run COCO evaluation on the last trained model
python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
```
## Training on your own datasets

Start by reading this  [blog post about the balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). 
It covers the process starting from annotating images to training to using the results in a sample application.
它涵盖了从标注图像到训练再到在示例应用程序中使用结果的过程。
In summary, to train the model on your own dataset you'll need to extend two classes:
总之，要在自己的数据集上训练模型，您需要扩展两个类：
`Config`  This class contains the default configuration. Subclass it and modify the attributes you need to change.
此类包含默认配置。 子类化并修改您需要更改的属性
`Dataset`  This class provides a consistent way to work with any dataset. It allows you to use new datasets for training without having to change the code of the model. It also supports loading multiple datasets at the same time, which is useful if the objects you want to detect are not all available in one dataset.
此类提供了使用任何数据集的一致方式。 它允许您使用新的数据集进行训练，而无需更改模型的代码。 它还支持同时加载多个数据集，这在您要检测的对象并非在一个数据集中全部可用时非常有用。
See examples in  `samples/shapes/train_shapes.ipynb`,  `samples/coco/coco.py`,  `samples/balloon/balloon.py`, and  `samples/nucleus/nucleus.py`.

## Training process 

 train for 160k iterations, starting from a learning rate of 0.02 and reducing it by 10 at 60k and 80k iterations. I  use bounding-box NMS with a threshold of 0.5.I use the Tesla P100 16g ,spent about 48 hours.
 从0.02的学习率开始进行160k迭代训练，然后在60k和80k迭代中将其减少10。 我使用阈值为0.5的边界框NMS。我使用了约48小时的Tesla P100 16g。
## The results shows

![humanestimtion](https://github.com/chrispolo/Keypoints-of-humanpose-with-Mask-R-CNN/blob/master/results/video1.png)
![humanestimation2](https://github.com/chrispolo/Keypoints-of-humanpose-with-Mask-R-CNN/blob/master/results/video2.png)
## Suggestions
You can put your problems in the issues, we can solve it together
Download the pretrained weights:
 https://pan.baidu.com/s/13V0n5m9ZU-ocbAks_GJwvw password: qx8f

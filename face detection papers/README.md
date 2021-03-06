- [人脸检测识别文献阅读总结](http://blog.csdn.net/zhangjunhit/article/details/78296135)
- [人脸检测综述](https://mp.weixin.qq.com/s?spm=a2c4e.11153940.blogcont591169.8.7b955beaoZ6r30&__biz=MzI1MjQ2OTQ3Ng==&mid=2247489529&idx=1&sn=112ef0c29991caa256c67cc23884ab4f&chksm=e9e21e72de9597641d4212af9f7bd70975e0572411eb4d5c068bafe43872b43f6b667904d708&scene=0#rd)
- [人脸检测论文汇总（不定期更新）](https://zhuanlan.zhihu.com/p/38512246)
# 人脸检测
## 定义
- 人脸检测（Face Detection），就是给一幅图像，找出图像中的所有人脸位置，通常用一个矩形框框起来，输入是一幅图像img，输出是若干个包含人脸的矩形框位置(x,y,w,h)
__在人脸检测方面，目前主流的方法是通用目标检测中的R-CNN等这类方法，Cascade CNN则是比较特异于人脸检测的方法，它将传统的滑动窗口方法与深度学习相结合，也取得了不亚于R-CNN系列方法的性能。人脸检测曾被认为是一个已经解决的问题，事实上并不是，在人脸分辨率极低、姿态很大、背光、偏光、极低照度等恶劣光照条件下，还是会有很多漏检。有鉴于此，去年出现了一个新的人脸检测Benchmark，应该会对人脸检测领域产生重要促进作用。__

## Benchmark
tinyface
wider-face

## 评价指标
- 召回率(recall)：detector能检测出来的人脸数量越多越好，由于每个图像中包含人脸的数量不一定，所以用检测出来的比例来衡量，这个指标就是召回率recall。detector检测出来的矩形框越接近人工标注的矩形框，说明检测结果越好，通常交并比IoU大于0.5就认为是检测出来了，所以 recall = 检测出来的人脸数量/图像中总人脸数量。
- 误检数(false positives)：detector也会犯错，可能会把其他东西认为是人脸，这种情况越少越好，我们用检测错误的绝对数量来表示，这个指标就是误检数false positives。与recall相对，detector检测出来的矩形框与任何人工标注框的IoU都小于0.5，则认为这个检测结果是误检，误检越少越好，比如FDDB上，论文中一般比较1000个或2000个误检时的召回率情况，工业应用中通常比较100或200个误检的召回率情况。
- 检测速度(speed)：是个算法都要比速度，人脸检测更不用说，detector检测一幅图像所用的时间越少越好，通常用帧率(frame-per-second，FPS)来表示。不过这里有点小问题，很多detector都是图像越小、图像中人脸越少、检测最小人脸越大，检测速度越快，需要注意不同论文的测试环境和测试图像可能不一样：测试图像，最常用的配置是VGA(640*480)图像检测最小人脸80*80给出速度，但都没有表明测试图像背景是否复杂，图像中有几个人脸（甚至是白底一人脸的图像测速度）；测试环境，差别就更大了，CPU有不同型号和主频，有多核多线程差异，GPU也有不同型号，等等。


## database
### FDDB
- [论文翻译-离散得分-连续得分](https://charlesnord.github.io/2017/04/07/FDDB/)
- 图像分辨率较小，所有图像的较长边缩放到450，也就是说所有图像都小于450*450，最小标注人脸20*20，包括彩色和灰度两类图像；
每张图像的人脸数量偏少，平均1.8人脸/图，绝大多数图像都只有一人脸；
- 数据集完全公开，published methods通常都有论文，大部分都开源代码且可以复现，可靠性高；unpublished methods没有论文没有代码，无法确认它们的训练集是否完全隔离，持怀疑态度最好，通常不做比较。（扔几张FDDB的图像到训练集，VJ也可以训练出很高的召回率。。需要考虑人品能不能抵挡住利益的诱惑）
有其他隔离数据集无限制训练再FDDB测试，和FDDB十折交叉验证两种，鉴于FDDB图像数量较少，近几年论文提交结果也都是无限制训练再FDDB测试方式，所以，如果要和published methods提交结果比较，请照做。山世光老师也说十折交叉验证通常会高1~3%。
- 从人脸检测的角度来看，在过去的几年里，学术界大多数还是在用FDDB做测试用的benchmark。目前，在这个共有2845幅图像、5171个人脸的数据集上，在共输出100个误检的情况下，用Fast R-CNN可以轻松取得90%以上的检测率或称召回率。工业界有些报道号称已经做到了95%，所以它基本上趋于饱和了。当然，值得特别注意的是，这个检测率在不少时候是有歧义的，有些团队报告的是10折平均的结果，有些报告的是一次性全部检测的结果，这两个结果是不可比的：10折平均的结果可能会偏高1-3个百分点。
- 结果有离散分数discROC和连续分数contROC两种，discROC仅关心IoU是不是大于0.5，contROC是IoU越大越好。鉴于大家都采用无限制训练加FDDB测试的方式，detector会继承训练数据集的标注风格，继而影响contROC，所以discROC比较重要，contROC看看就行了，不用太在意。
### Wider Face
- 2016年人脸检测领域的一个重要变化是出现了一个新的Benchmark：香港中文大学贡献了一个规模更大、数据变化更丰富的新数据集——Wider Face。其中包括1.6万测试图像，共19.4万个标注人脸。更重要的是，如上图所示，数据集中的人脸有大小、姿态、光照、遮挡以及表情等各方面非常复杂的变化。特别的，其中50%的人脸高度小于50个像素，甚至大量高度小于20个像素的Tiny face。
- Wider Face将测试图像分为“难”、“中”、“易”三种不同的难度等级。从目前State of the art方法的检测曲线不难看出，在最“难”的测试子集上，目前只能做到80%的检测率和80%的精度，对检测任务而言，这是相当低的结果了。可见，在该数据集上，现有方法的性能在“难”等级下还有非常长的路可以走。

## 性能比较

![深度学习人脸检测-FDDB](https://github.com/geyongtao/deep-face-pipeline/blob/master/face%20detection%20papers/pictures/performace-FDDB.jpg)


- 级联CNN系列，有CNN Cascade, FaceCraft, MTNN, ICC-CNN，这一系列是深度学习方法中速度最快的，CPU都在10 FPS以上，级联CNN系列优化后轻松可以在CPU上实时，全面优化后的fastMTCNN甚至可以在ARM上跑起来；
- Faster R-CNN系列，性能可以做到极高，但速度都很慢，甚至不能在GPU上实时；
- SSD/RPN系列：有SSH和SFD，都是目前FDDB和WIDER FACE上的最高水平，性能水平与Faster R-CNN系列不相上下，同时也可以保持GPU实时速度，SFD的简化版FaceBoxes甚至可以CPU上实时，极有潜力上ARM。



【Face Detection】
<ul>
<li>
A Convolutional Neural Network Cascade for Face Detection, CVPR, 2015
</ul></li>
  
<ul>
<li>
Joint Training of Cascaded CNN for Face Detection, CVPR, 2016. <a href="http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Qin_Joint_Training_of_CVPR_2016_paper.pdf">[Paper]</a></li>
</ul></li>

<ul>
<li>
WIDER FACE: A Face Detection Benchmark, CVPR, 2016. <a href="http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_WIDER_FACE_A_CVPR_2016_paper.pdf">[Paper]</a></li>
</ul></li>

<ul><li>
Face Detection with End-to-End Integration of a ConvNet and a 3D Model, ECCV, 2016.
</ul></li>
  
<ul><li>
 Recurrent Scale Approximation for Object Detection in CNN, ICCV, 2017.
 </ul></li>
 
 <ul><li>
  finding tiny face, CVPR, 2017. <a href="https://www.cs.cmu.edu/~peiyunh/tiny/">[Homepage]</a></li>
  </ul></li>
  
 <ul><li>
 自动化所_FaceBoxes: A CPU Real-time Face Detector with High Accuracy <a href="https://github.com/zeusees/FaceBoxes">[reimplenment]</a></li>
</ul></li>
 
 <ul><li> 
 华为_Face Detection Using Improved Faster RCNN <a href="https://arxiv.org/abs/1802.02142">[arxiv]
 </ul></li>
  
 <ul><li>
 百度_PyramidBox: A Context-assisted Single Shot Face Detector <a href="https://arxiv.org/abs/1803.07737?context=cs">[arxiv]</a></li>
 </ul></li>
 
 
 <ul><li>
北航_Beyond Trade-off: Accelerate FCN-based Face Detector with Higher Accuracy, CVPR, 2018. <a href="https://arxiv.org/abs/1804.05197">[arxiv]</a></li>
</ul></li>
  

<ul><li>
Finding Tiny Faces in the Wild with Generative Adversarial Network, CVPR, 2018. <a href="https://ivul.kaust.edu.sa/Pages/pub-tiny-faces.aspx">[Homepage]</a></li>
</ul></li>

<ul><li>
CMU_Seeing Small Faces from Robust Anchor’s Perspective, CVPR, 2018. <a href="https://arxiv.org/abs/1802.09058">[arxiv]</a></li>
</ul></li>

## R-CNN based

- Learned-Miller. Face detection with the faster r-cnn, FG, 2017.
- Cms-rcnn: contextual multi-scale region-based cnn for unconstrained face detection. Deep Learning for Biometrics, 2017.
   


## 人脸检测数据集
数据库| 描述| 用途| 获取方法
--------------|-------------|------------|------------
<b>FDDB</b> | 2845张图片中的5171张脸 | 标准人脸检测评测集 | [链接](http://vis-www.cs.umass.edu/fddb/)
<b>IJB-A</b> | |人脸识别，人脸检测 |[链接](http://www.nist.gov/itl/iad/ig/ijba_request.cfm)
<b>Caltech10k Web Faces</b> |10k+人脸，提供双眼和嘴巴的坐标位置 | 人脸点检测| [链接](http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/#Description)

## 人脸检测数据集描述
### MALF
MALF 数据集全称 Multi-Attribute Labelled Faces ，是一个大型数据库，被设计用于野外细粒度人脸检测评估。网站提供了：

- 数据库的描述、标注和评估规则
- 如何下载数据库进行评估
- 如何向该网站提交你的检测结果，和他人进行对比
- 所有算法的性能对比
- 数据库包含 5,250 张图片和 11,931 张标注的人脸，图片都来自于互联网。

### FDDB
FDDB 数据集全称 Face Detection Data Set and Benchmark ，是一个用于研究无约束条件下人脸检测问题的数据集。数据库包含 2，845 张图片和 5，171 张标注的人脸，图片都来自于 Faces in the Wild 数据集。

### AFW
AFW 数据集全称 Annotated Faces in the Wild ，AFW数据集是使用Flickr（雅虎旗下图片分享网站）图像建立的人脸图像库，包含**205个图像**，其中有**473个标记的人脸**。对于每一个人脸都包含一个长方形边界框，**6个地标**和相关的姿势角度。数据库虽然不大，额外的好处是作者给出了其2012 CVPR的论文和程序以及训练好的模型。

### IJBA-A
IJBA-A 数据集全称IARPA Janus Benchmark A，是由 IARPA (the Intelligence Advanced Research Projects Activity)公开的数据集，包含24327个图像和49759个人脸，可以用于人脸的检测和识别。数据集需要进行申请，经过NIST(美国国家标准技术研究所)允许同意后，才能获取到。

### WIDER FACE
WIDER FACE 数据集是一个人脸检测基准数据库，其中的图片是从已经公布的WIDER数据集上挑选出来的。它包含**32203**个图像和**393703**个人脸图像，在尺度，姿势，闭塞，表达等方面表现出了大的变化。WIDER FACE 数据集基于**61个事件**类型进行组织的。对于每一个事件类型，随机选取40%的数据作为训练集、10%的数据集作为验证集、50%的数据作为测试集。同时，WIDER FACE 数据集采用和 PASCAL VOC数据集相同的评估度量策略。 和 MALF、Caltech 数据集一样, 并不公开测试集的 ground truth (考试的时候，当然不公布正确答案啦)。用户被要求提交最终的预测文件，从而通过网站进行评估，给出结果。 


## 开源项目
[yolo v3 face detection](https://www.jianshu.com/p/8214d947e031)

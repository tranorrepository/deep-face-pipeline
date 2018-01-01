# Face-Recognition-Papers
awesome deep learning papers for face recognition

# Content
  *   [Popular_Architecture](#popular_architecture)

      *   [DeepFace](#deepface)
      *   [DeepID](#deepid)
      *   [FaceNet](#facenet)
      *   [VIPFaceNet](#vipfacenet)
      *   [WebFace](#webface)
      *   [VGGFace](#vggface)
      *   [BaiduFace](#baiduface)
      *   [Face++](#face++)
      *   [OpenFace](#openface)
      *   [Pruning_Network](#pruning_network)
      *   [CenterFace](#centerface)
      *   [MegaFace](#megaface)
      *   [ModelID](#modelid)
   
  *   [Loss_function](#loss_function)
  *   [Dataset](#dataset)
  *   [Video_Based_Method](#video_based_method)
  *   [Process](#process)
      *   [Face_alignment](#face_alignment)
      *   [Face_verfication](face_verfication)
  
  *   [Resources](#resources)
  
  
* * *

   
# Popular_Architecture

## DeepFace
- [DeepFace: Closing the Gap to Human-Level Performance in Face Verification]
- Convs followed by locally connected, followed by fully connected
(https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf) [Yaniv Taigman et al., 2014]

- [Web-Scale Training for Face Identification](https://arxiv.org/abs/1406.5266) [Yaniv Taigman et al., 2015]

- intro: CVPR 2014. Facebook AI Research
- paper: [https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)
- slides: [http://valse.mmcheng.net/ftp/20141126/MingYang.pdf](http://valse.mmcheng.net/ftp/20141126/MingYang.pdf)
- github: [https://github.com/RiweiChen/DeepFace](https://github.com/RiweiChen/DeepFace)

## DeepID
- They use verification and identification signals to train the network. Afer each convolutional layer there is an identity layer connected to the supervisory signals in order to train each layer closely (on top of normal backprop)
- [Deep Learning Face Representation from Predicting 10,000 Classes](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf) [Yi Sun et al., 2014]

- [DeepID2: Deep Learning Face Representation by Joint Identification-Verification](https://arxiv.org/abs/1406.4773) [Yi Sun et al., 2014]

- [DeepID2+: Deeply learned face representations are sparse, selective, and robust](https://arxiv.org/abs/1412.1265) [Yi Sun et al., 2014]

- [DeepID3: Face Recognition with Very Deep Neural Networks](https://arxiv.org/abs/1502.00873) [Yi Sun et al., 2015]


- intro: CVPR 2014
- paper: [http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)
- github: [https://github.com/stdcoutzyx/DeepID_FaceClassify](https://github.com/stdcoutzyx/DeepID_FaceClassify)

**基于Caffe的DeepID2实现**

- 1. [http://www.miaoerduo.com/deep-learning/%E5%9F%BA%E4%BA%8Ecaffe%E7%9A%84deepid2%E5%AE%9E%E7%8E%B0%EF%BC%88%E4%B8%8A%EF%BC%89.html](http://www.miaoerduo.com/deep-learning/%E5%9F%BA%E4%BA%8Ecaffe%E7%9A%84deepid2%E5%AE%9E%E7%8E%B0%EF%BC%88%E4%B8%8A%EF%BC%89.html)
- 2. [http://www.miaoerduo.com/deep-learning/%E5%9F%BA%E4%BA%8Ecaffe%E7%9A%84deepid2%E5%AE%9E%E7%8E%B0%EF%BC%88%E4%B8%AD%EF%BC%89.html](http://www.miaoerduo.com/deep-learning/%E5%9F%BA%E4%BA%8Ecaffe%E7%9A%84deepid2%E5%AE%9E%E7%8E%B0%EF%BC%88%E4%B8%AD%EF%BC%89.html)
- 3. [http://www.miaoerduo.com/deep-learning/%E5%9F%BA%E4%BA%8Ecaffe%E7%9A%84deepid2%E5%AE%9E%E7%8E%B0%EF%BC%88%E4%B8%8B%EF%BC%89.html](http://www.miaoerduo.com/deep-learning/%E5%9F%BA%E4%BA%8Ecaffe%E7%9A%84deepid2%E5%AE%9E%E7%8E%B0%EF%BC%88%E4%B8%8B%EF%BC%89.html)

**DeepID2+视频**
- video: [http://research.microsoft.com/apps/video/?id=260023](http://research.microsoft.com/apps/video/?id=260023)
- mirror: [http://pan.baidu.com/s/1boufl3x](http://pan.baidu.com/s/1boufl3x)

## FaceNet
- They use a triplet loss with the goal of keeping the L2 intra-class distances low and inter-class distances high 
- [FaceNet: A Unified Embedding for Face Recognition and Clustering](http://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html) [Florian Schroff et al., 2015]

- intro: Google Inc. CVPR 2015
- arxiv: [http://arxiv.org/abs/1503.03832](http://arxiv.org/abs/1503.03832)
- github(Tensorflow): [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)
- github(Caffe): [https://github.com/hizhangp/triplet](https://github.com/hizhangp/triplet)

**Real time face detection and recognition**

- intro: Real time face detection and recognition base on opencv/tensorflow/mtcnn/facenet
- github: [https://github.com/shanren7/real_time_face_recognition](https://github.com/shanren7/real_time_face_recognition)

- - -

**Targeting Ultimate Accuracy: Face Recognition via Deep Embedding**

- intro: CVPR 2015
- arxiv: [http://arxiv.org/abs/1506.07310](http://arxiv.org/abs/1506.07310)

**Learning Robust Deep Face Representation**

- arxiv: [https://arxiv.org/abs/1507.04844](https://arxiv.org/abs/1507.04844)

**A Light CNN for Deep Face Representation with Noisy Labels**

- arxiv: [https://arxiv.org/abs/1511.02683](https://arxiv.org/abs/1511.02683)
- github: [https://github.com/AlfredXiangWu/face_verification_experiment](https://github.com/AlfredXiangWu/face_verification_experiment)

**Pose-Aware Face Recognition in the Wild**

- paper: [www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Masi_Pose-Aware_Face_Recognition_CVPR_2016_paper.pdf](www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Masi_Pose-Aware_Face_Recognition_CVPR_2016_paper.pdf)

**Triplet Probabilistic Embedding for Face Verification and Clustering**

- intro: Oral Paper in BTAS 2016; NVIDIA Best paper Award
- arxiv: [https://arxiv.org/abs/1604.05417](https://arxiv.org/abs/1604.05417)
- github(Keras): [https://github.com/meownoid/face-identification-tpe](https://github.com/meownoid/face-identification-tpe)

**Recurrent Regression for Face Recognition**

- arxiv: [http://arxiv.org/abs/1607.06999](http://arxiv.org/abs/1607.06999)

**A Discriminative Feature Learning Approach for Deep Face Recognition**

- intro: ECCV 2016
- intro: center loss
- paper: [http://ydwen.github.io/papers/WenECCV16.pdf](http://ydwen.github.io/papers/WenECCV16.pdf)
- github: [https://github.com/ydwen/caffe-face](https://github.com/ydwen/caffe-face)
- github: [https://github.com/pangyupo/mxnet_center_loss](https://github.com/pangyupo/mxnet_center_loss)

**How Image Degradations Affect Deep CNN-based Face Recognition?**

- arxiv: [http://arxiv.org/abs/1608.05246](http://arxiv.org/abs/1608.05246)

## VIPLFaceNet

**VIPLFaceNet: An Open Source Deep Face Recognition SDK**
**SeetaFace Engine

- arxiv: [http://arxiv.org/abs/1609.03892](http://arxiv.org/abs/1609.03892)

**SeetaFace Engine**

- intro: SeetaFace Engine is an open source C++ face recognition engine, which can run on CPU with no third-party dependence.
- github: [https://github.com/seetaface/SeetaFaceEngine](https://github.com/seetaface/SeetaFaceEngine)

**A Discriminative Feature Learning Approach for Deep Face Recognition**

- intro: ECCV 2016
- paper: [http://ydwen.github.io/papers/WenECCV16.pdf](http://ydwen.github.io/papers/WenECCV16.pdf)

**Sparsifying Neural Network Connections for Face Recognition**

- paper: [http://www.ee.cuhk.edu.hk/~xgwang/papers/sunWTcvpr16.pdf](http://www.ee.cuhk.edu.hk/~xgwang/papers/sunWTcvpr16.pdf)

**Range Loss for Deep Face Recognition with Long-tail**

- arxiv: [https://arxiv.org/abs/1611.08976](https://arxiv.org/abs/1611.08976)

**Hybrid Deep Learning for Face Verification**

- intro: TPAMI 2016. CNN+RBM
- paper: [http://www.ee.cuhk.edu.hk/~xgwang/papers/sunWTpami16.pdf](http://www.ee.cuhk.edu.hk/~xgwang/papers/sunWTpami16.pdf)

**Towards End-to-End Face Recognition through Alignment Learning**

- intro: Tsinghua University
- arxiv: [https://arxiv.org/abs/1701.07174](https://arxiv.org/abs/1701.07174)

**Multi-Task Convolutional Neural Network for Face Recognition**

- arxiv: [https://arxiv.org/abs/1702.04710](https://arxiv.org/abs/1702.04710)

**NormFace: L2 Hypersphere Embedding for Face Verification**

- arxiv: [https://arxiv.org/abs/1704.06369](https://arxiv.org/abs/1704.06369)
- github: [https://github.com/happynear/NormFace](https://github.com/happynear/NormFace)

**SphereFace: Deep Hypersphere Embedding for Face Recognition**

- intro: CVPR 2017
- arxiv: [http://wyliu.com/papers/LiuCVPR17.pdf](http://wyliu.com/papers/LiuCVPR17.pdf)
- github: [https://github.com/wy1iu/sphereface](https://github.com/wy1iu/sphereface)
- demo: [http://v-wb.youku.com/v_show/id_XMjk3NTc1NjMxMg==.html](http://v-wb.youku.com/v_show/id_XMjk3NTc1NjMxMg==.html)

**L2-constrained Softmax Loss for Discriminative Face Verification**

[https://arxiv.org/abs/1703.09507](https://arxiv.org/abs/1703.09507)

**Low Resolution Face Recognition Using a Two-Branch Deep Convolutional Neural Network Architecture**

- intro: Amirkabir University of Technology & MIT
- arxiv: [https://arxiv.org/abs/1706.06247](https://arxiv.org/abs/1706.06247)

**Enhancing Convolutional Neural Networks for Face Recognition with Occlusion Maps and Batch Triplet Loss**

[https://arxiv.org/abs/1707.07923](https://arxiv.org/abs/1707.07923)

**Model Distillation with Knowledge Transfer in Face Classification, Alignment and Verification**

[https://arxiv.org/abs/1709.02929](https://arxiv.org/abs/1709.02929)

**Improving Heterogeneous Face Recognition with Conditional Adversarial Networks**

[https://arxiv.org/abs/1709.02848](https://arxiv.org/abs/1709.02848)

**Face Sketch Matching via Coupled Deep Transform Learning**

- intro: ICCV 2017
- arxiv: [https://arxiv.org/abs/1710.02914](https://arxiv.org/abs/1710.02914)


## WebFace
- [Learning Face Representation from Scratch](https://arxiv.org/pdf/1411.7923.pdf) [Dong Yi et al., 2014]

- [A Lightened CNN for Deep Face Representation](https://pdfs.semanticscholar.org/d4e6/69d5d35fa0ca9f8d9a193c82d4153f5ffc4e.pdf) [[Xiang Wu et al., 2015]

- [A Light CNN for Deep Face Representation with Noisy Labels](https://arxiv.org/abs/1511.02683) [Xiang Wu et al., 2017]

## VGGFace
- [Deep Face Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) [Omkar M. Parkhi et al., 2015]

## BaiduFace
- [Targeting Ultimate Accuracy: Face Recognition via Deep Embedding](https://arxiv.org/abs/1506.07310) [Jingtuo Liu et al., 2015]

## Face++
- [Naive-Deep Face Recognition: Touching the Limit of LFW Benchmark or Not?](https://arxiv.org/abs/1501.04690) [Erjin Zhou et al., 2015]

## OpenFace
- [OpenFace: A general-purpose face recognition library with mobile applications](https://cmusatyalab.github.io/openface/) [Brandon Amos et al., 2016]

## Pruning_Network
- [DSD: Dense-Sparse-Dense Training for Deep Neural Networks](https://arxiv.org/abs/1607.04381) [Song Han et al., 2017]

- [Pruning Convolutional Neural Networks for Resource Efficient Transfer Learning](https://arxiv.org/abs/1611.06440) [Pavlo Molchanov et al., 2017]

- [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626) [Song Han et al., 2016]

## CenterFace
- [A Discriminative Feature Learning Approach for Deep Face Recognition](http://ydwen.github.io/papers/WenECCV16.pdf) [Yandong Wen et al., 2016]

## MegaFace
- [MegaFace: A Million Faces for Recognition at Scale](https://arxiv.org/abs/1505.02108) [D. Miller et al., 2016]

- [The MegaFace Benchmark: 1 Million Faces for Recognition at Scale](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kemelmacher-Shlizerman_The_MegaFace_Benchmark_CVPR_2016_paper.pdf) [Ira Kemelmacher-Shlizerman et al., 2016]

## MobileID

**MobileID: Face Model Compression by Distilling Knowledge from Neurons**

- intro: AAAI 2016 Oral. CUHK
- intro: MobileID is an extremely fast face recognition system by distilling knowledge from DeepID2
- project page: [http://personal.ie.cuhk.edu.hk/~lz013/projects/MobileID.html](http://personal.ie.cuhk.edu.hk/~lz013/projects/MobileID.html)
- paper: [http://personal.ie.cuhk.edu.hk/~pluo/pdf/aaai16-face-model-compression.pdf](http://personal.ie.cuhk.edu.hk/~pluo/pdf/aaai16-face-model-compression.pdf)
- github: [https://github.com/liuziwei7/mobile-id](https://github.com/liuziwei7/mobile-id)

## Joint Bayesian
- [Bayesian Face Revisited: A Joint Formulation](http://www.jiansun.org/papers/ECCV12_BayesianFace.pdf) [Dong Chen et al., 2012]

- [A Practical Transfer Learning Algorithm for Face Verification](http://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Cao_A_Practical_Transfer_2013_ICCV_paper.pdf) [Xudong Cao et al., 2013]

# Loss_function
- [Beyond triplet loss: a deep quadruplet network for person re-identification](https://arxiv.org/pdf/1704.01719.pdf) [Weihua Chen et al., 2017]

- [Range Loss for Deep Face Recognition with Long-tail](https://arxiv.org/abs/1611.08976) [Xiao Zhang et al., 2016]
- [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063) [Weiyang Liu al., 2017](A softmax loss)



# Dataset

1. [CASIA WebFace Database](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html). 10,575 subjects and 494,414 images
2. [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/).13,000 images and 5749 subjects
3. [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/) 202,599 images and 10,177 subjects. 5 landmark locations, 40 binary attributes.
4. [MSRA-CFW](http://research.microsoft.com/en-us/projects/msra-cfw/). 202,792 images and 1,583 subjects.
5. [MegaFace Dataset](http://megaface.cs.washington.edu/) 1 Million Faces for Recognition at Scale
690,572 unique people
(https://arxiv.org/pdf/1512.00596.pdf) CVPR 2016
6. [FaceScrub](http://vintage.winklerbros.net/facescrub.html). A Dataset With Over 100,000 Face Images of 530 People.
7. [FDDB](http://vis-www.cs.umass.edu/fddb/).Face Detection and Data Set Benchmark. 5k images.
8. [AFLW](https://lrs.icg.tugraz.at/research/aflw/).Annotated Facial Landmarks in the Wild: A Large-scale, Real-world Database for Facial Landmark Localization. 25k images.
9. [AFW](http://www.ics.uci.edu/~xzhu/face/). Annotated Faces in the Wild. ~1k images.
10.[3D Mask Attack Dataset](https://www.idiap.ch/dataset/3dmad). 76500 frames of 17 persons using Kinect RGBD with eye positions (Sebastien Marcel)
11. [Audio-visual database for face and speaker recognition](https://www.idiap.ch/dataset/mobio).Mobile Biometry MOBIO http://www.mobioproject.org/
12. [BANCA face and voice database](http://www.ee.surrey.ac.uk/CVSSP/banca/). Univ of Surrey
13. [Binghampton Univ 3D static and dynamic facial expression database](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html). (Lijun Yin, Peter Gerhardstein and teammates)
14. [The BioID Face Database](https://www.bioid.com/About/BioID-Face-Database). BioID group
15. [Biwi 3D Audiovisual Corpus of Affective Communication](http://www.vision.ee.ethz.ch/datasets/b3dac2.en.html).  1000 high quality, dynamic 3D scans of faces, recorded while pronouncing a set of English sentences.
16. [Cohn-Kanade AU-Coded Expression Database](http://www.pitt.edu/~emotion/ck-spread.htm).  500+ expression sequences of 100+ subjects, coded by activated Action Units (Affect Analysis Group, Univ. of Pittsburgh.
17. [CMU/MIT Frontal Faces ](http://cbcl.mit.edu/software-datasets/FaceData2.html). Training set:  2,429 faces, 4,548 non-faces; Test set: 472 faces, 23,573 non-faces.
18. [AT&T Database of Faces](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) 400 faces of 40 people (10 images per people)
19. [MS-Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/) [Yandong Guo et al., 2016]


## Feature Normalization Method
- [L2-constrained Softmax Loss for Discriminative Face Verification](https://arxiv.org/abs/1703.09507v2) [Rajeev Ranjan al., 2017]

- [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063) [Weiyang Liu al., 2017]

- [DeepVisage: Making face recognition simple yet with powerful generalization skills](https://arxiv.org/abs/1703.08388) [Abul Hasnat al., 2017]


## Face Recognition
- [Learning Deep Representation for Imbalanced Classification](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Huang_Learning_Deep_Representation_CVPR_2016_paper.pdf) CVPR 2016
- [Latent Factor Guided Convolutional Neural Networks for Age-Invariant Face Recognition](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Wen_Latent_Factor_Guided_CVPR_2016_paper.pdf) CVPR 2016
- [Sparsifying Neural Network Connections for Face Recognition](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Sun_Sparsifying_Neural_Network_CVPR_2016_paper.pdf) CVPR 2016
- [Pose-Aware Face Recognition in the Wild](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Masi_Pose-Aware_Face_Recognition_CVPR_2016_paper.pdf) CVPR 2016
- [Do We Really Need to Collect Millions of Faces for Effective Face Recognition?](http://arxiv.org/pdf/1603.07057v2.pdf) (2016)
   - A method for training data augumentation is porposed as alternative to the manual harvesting and labeling of millions (up to 200M!) of face images recently used to achieve the top results in LFW (Google, Facebook etc)
   - Their data augmentation goes beyond traditional techniques known to work well for deep learning such oversampling by cropping and shifting multiple times each original image, mirroring, rotating, etc. Insted they use domain specific data augumentation and generate new samples by varying pose, shape, and expression.
- Baidu's [Targeting Ultimate Accuracy: Face Recognition via Deep Embedding](http://arxiv.org/pdf/1506.07310v4.pdf) (2015)
   - Similar approach to [FaceNet](http://arxiv.org/pdf/1503.03832v3.pdf)
   - Multi-patch deep CNN followed by deep metric learning using triplet loss
   - Increasing the number of images from 150K to 1.2M reduces the error rate from 3.1% to 0.87%
   - Increasing the number of patches from 1 to 7 reduces the error rate from 0.87% to 0.32%. Increasing the number of patches further does not improve the results, actually makes the error rate slightly worse.
- Google's [FaceNet: A Unified Embedding for Face Recognition and Clustering](http://arxiv.org/pdf/1503.03832v3.pdf) (2015)
   - They use a triplet loss with the goal of keeping the L2 intra-class distances low and inter-class distances high
- [Network In Network](http://arxiv.org/pdf/1312.4400v3.pdf) (2014)
   - "In NIN, the GLM is replaced with a ”micro network” structure which is a general nonlinear function approximator"
   - The fully connected layers (classifiers) at the end of the convolution layers (feature extractors) are replaced by a global average pooling layer, i.e., the last mlpconv layer produces one feature map per class which is then followed by a softmax layer
      - This approach has two advantages: 1) It has better generalization properties than the traditional fully connected layer which often suffers from overfitting (which is normally sorted using droppout) 2) Each output feature map is basically a confidence map for one class which is very intuintive and meaningfull
   - The micro network chosen in this paper is a multilayer perceptron
   - Has advantages over the traditional fully connected layers at the output
- [CP-mtML: Coupled Projection multi-task Metric Learning for Large Scale Face Retrieval](http://arxiv.org/pdf/1604.02975v1.pdf) CVPR 2016

## Face_alignment
- [Mnemonic Descent Method: A recurrent process applied for end-to-end face alignment](http://ibug.doc.ic.ac.uk/media/uploads/documents/trigeorgis2016mnemonic.pdf) CVPR 2016
- [Large-pose Face Alignment via CNN-based Dense 3D Model Fitting](http://cvlab.cse.msu.edu/pdfs/Jourabloo_Liu_CVPR2016.pdf) CVPR 2016

## Face_verfication
## Fast face recognition and verification
- [Real-Time Face Identification via CNN and Boosted Hashing Forest](http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w4/papers/Vizilter_Real-Time_Face_Identification_CVPR_2016_paper.pdf) CVPR 2016







**Deep Face Recognition**

- intro: BMVC 2015
- paper: [http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
- homepage: [http://www.robots.ox.ac.uk/~vgg/software/vgg_face/](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)
- github(Keras): [https://github.com/rcmalli/keras-vggface](https://github.com/rcmalli/keras-vggface)




# Video_Based_Method

**Attention-Set based Metric Learning for Video Face Recognition**

[https://arxiv.org/abs/1704.03805](https://arxiv.org/abs/1704.03805)

## Projects

**Using MXNet for Face-related Algorithm**

- github: [https://github.com/tornadomeet/mxnet-face](https://github.com/tornadomeet/mxnet-face)


### OpenFace

**OpenFace: Face Recognition with Deep Neural Networks**

- homepage: [http://cmusatyalab.github.io/openface/](http://cmusatyalab.github.io/openface/)
- github: [https://github.com/cmusatyalab/openface](https://github.com/cmusatyalab/openface)
- github: [https://github.com/aybassiouny/OpenFaceCpp](https://github.com/aybassiouny/OpenFaceCpp)

**OpenFace 0.2.0: Higher accuracy and halved execution time**

- homepage: [http://bamos.github.io/2016/01/19/openface-0.2.0/](http://bamos.github.io/2016/01/19/openface-0.2.0/)

**OpenFace: A general-purpose face recognition library with mobile applications**

- paper: [http://reports-archive.adm.cs.cmu.edu/anon/anon/usr0/ftp/2016/CMU-CS-16-118.pdf](http://reports-archive.adm.cs.cmu.edu/anon/anon/usr0/ftp/2016/CMU-CS-16-118.pdf)

**FaceVerification: An Experimental Implementation of Face Verification, 96.8% on LFW**

- github: [https://github.com/happynear/FaceVerification](https://github.com/happynear/FaceVerification)

**OpenFace: an open source facial behavior analysis toolkit**

![](https://raw.githubusercontent.com/TadasBaltrusaitis/OpenFace/master/imgs/multi_face_img.png)

- intro: a state-of-the art open source tool intended for facial landmark detection, head pose estimation, 
facial action unit recognition, and eye-gaze estimation.
- github: [https://github.com/TadasBaltrusaitis/OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)

**InsightFace**

- intro: Face Recognition Project on MXnet
- arxiv: [https://github.com//deepinsight/insightface](https://github.com//deepinsight/insightface)

## Resources

**Face-Resources**

- github: [https://github.com/betars/Face-Resources](https://github.com/betars/Face-Resources)

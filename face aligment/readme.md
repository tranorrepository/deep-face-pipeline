## Face Aligment
------------------------------
- 主动形状模型(Acitve Shape Model,ASM)[Cootes, BMVC/1992]
- 主动表现模型(Active Appearance Model, AAM)[Cootes, ECCV/1998]
- 局部约束模型(Constrained Local Model, CLM)[Cristinacce, BMVC/2006]
> Cristinacce D, Cootes T F. Feature Detection and Tracking with Constrained Local Models[C]// British Machine Vision Conference 2006, Edinburgh, Uk, September. DBLP, 2006:929-938.

- 级联形状回归(Cascade Shape Regression, CSR) 2010-至今
- Cascaded Pose Regression(CPR)[Dollar ,CVPR/2010
- Supervised Descent Method(SDM) [Xiong, CVPR/2013]
- Explict Shape Regression(ESR)[Cao, IJCV/2014]
- Cascaded Deep Neural Networks[Zhang, ECCV/2014]



- [Face Alignment Across Large Poses: A 3D Solution](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)[CVPR2016]
```
method：
1.构成一个人脸需要参数集p。构建一个与pk（表示p的第k次迭代）相关的feature ：PNCC，把PNCC和图片输入CNN多次迭代并更新p以后，生成了所需的参数集p

2.构建两个3d模型，一个在脸部区域用MFF方法，另一个在非脸部区域使用3d meshing方法，然后两者结合构成模型。然后旋转模型，产生侧脸图片。
```

- [Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network](https://github.com/YadiraF/PRNet)

- [How far are we from solving the 2D \& 3D Face Alignment problem](https://github.com/1adrianb/face-alignment)[ICCV 2017]


<ul>
<li>Deep Convolutional Network Cascade for Facial Point Detection, CVPR, 2013.
</ul></li>
       
<ul>       
 <li> Facial Landmark Detection by Deep Multi-task Learning, ECCV, 2014. <a href="https://github.com/zhzhanp/TCDCN-face-alignment">[Code]
</ul></li>
       
<ul>
<li>Face Alignment Across Large Poses: A 3D Solution, CVPR, 2016. <a href="http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Face_Alignment_Across_CVPR_2016_paper.pdf">[Paper]</a></li>
</ul></li>

<ul>
<li>Unconstrained Face Alignment via Cascaded Compositional Learning, CVPR, 2016. <a href="http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Unconstrained_Face_Alignment_CVPR_2016_paper.pdf">[Paper]</a></li>
</ul></li>

<ul>
<li>Occlusion-Free Face Alignment: Deep Regression Networks Coupled With De-Corrupt AutoEncoders, CVPR, 2016. <a href="http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Occlusion-Free_Face_Alignment_CVPR_2016_paper.pdf">[Paper]</a></li>
</ul></li>

<ul>
<li>Mnemonic Descent Method: A Recurrent Process Applied for End-To-End Face Alignment, CVPR, 2016. <a href="http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Trigeorgis_Mnemonic_Descent_Method_CVPR_2016_paper.pdf">[Paper]</a></li>
</ul></li>

<ul>
<li>Large-Pose Face Alignment via CNN-Based Dense 3D Model Fitting, CVPR, 2016. <a href="http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Jourabloo_Large-Pose_Face_Alignment_CVPR_2016_paper.pdf">[Paper]</a></li>
</ul></li>

<ul><li>
A Deep Regression Architecture with Two-Stage Re-initialization for High Performance Facial Landmark Detection, CVPR, 2017.
</ul></li>

<ul><li>
Supervision-by-Registration: An Unsupervised Approach to Improve the Precision of Facial Landmark Detectors, CVPR, 2018.
      <a href="https://github.com/facebookresearch/supervision-by-registration">[code]</a></li>
</ul></li>
       
 <ul><li>
Fengzhenhua_Supervision-by-Registration: An Unsupervised Approach to Improve the Precision of Facial Landmark Detectors, CVPR, 2018.
      <a href="https://github.com/FengZhenhua/Wing-Loss">[code]</a></li>
</ul></li>

 <ul><li>
Yuting Zhang_Unsupervised Discovery of Object Landmarks as Structural Representations, CVPR, 2018.
       <a href="https://github.com/YutingZhang/lmdis-rep">[code]</a></li>
</ul></li>

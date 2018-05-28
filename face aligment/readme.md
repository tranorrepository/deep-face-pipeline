- [Face Alignment Across Large Poses: A 3D Solution](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)[CVPR2016]
```
method：
1.构成一个人脸需要参数集p。构建一个与pk（表示p的第k次迭代）相关的feature ：PNCC，把PNCC和图片输入CNN多次迭代并更新p以后，生成了所需的参数集p

2.构建两个3d模型，一个在脸部区域用MFF方法，另一个在非脸部区域使用3d meshing方法，然后两者结合构成模型。然后旋转模型，产生侧脸图片。
```

- [Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network](https://github.com/YadiraF/PRNet)

- [How far are we from solving the 2D \& 3D Face Alignment problem](https://github.com/1adrianb/face-alignment)[ICCV 2017]

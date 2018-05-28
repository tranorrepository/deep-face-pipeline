- [CVPR2016: Face Alignment Across Large Poses: A 3D Solution](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
method：
1.构成一个人脸需要参数集p。构建一个与pk（表示p的第k次迭代）相关的feature ：PNCC，把PNCC和图片输入CNN多次迭代并更新p以后，生成了所需的参数集p

2.构建两个3d模型，一个在脸部区域用MFF方法，另一个在非脸部区域使用3d meshing方法，然后两者结合构成模型。然后旋转模型，产生侧脸图片。

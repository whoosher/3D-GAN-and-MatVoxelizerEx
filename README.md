# 3D-GAN-and-MatVoxelizer
This contents contain Three dimensional Generative Adversarial Network(3D-GAN) in Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling (NeualPS2016).

For your custom data training, Just use Sample_Run.py with adjusting data shape and type.

In this contents, the ShapeNet 3D object shape data which is extended with .mat file (the air plane example is used).

The model input datatype is binvox, which is consist of binary value for voxel(64x64x64). Input shape is (n,1,1,64,64,64)

Also, the example of voxelizer for matobject (MatVoxelizer) is developed to train 3D-GAN in EX_MatVoxelizer.py. 


![그림1](https://user-images.githubusercontent.com/62490138/174087387-00c5975e-cecd-4dea-ad9f-8403cfc24680.png)




Reference)

1. Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling (NeualPS2016): Wu, J., Zhang, C., Xue, T., Freeman, B., & Tenenbaum, J. (2016). Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. Advances in neural information processing systems, 29.

2. Pytorch DCGAN: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import os
import glob
import numpy as np
import scipy.io as io
import scipy.ndimage as nd



def EX_airplane (dir_path,object_ratio=float(1.0)):

    result = list()
    airplanes = glob.glob(dir_path + '/*.mat')
    file_list = airplanes[0:int(object_ratio * len(airplanes))]

    for idx,data in enumerate(file_list):

        # Voxel화 작업
        test_image_airplane = data
        voxels = io.loadmat(test_image_airplane)['instance']
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
        result.append([[voxels]])

    volumes = np.asarray(result, dtype=float)
    return volumes

# airplanes = r'D:\3DShapeNets\volumetric_data\airplane\30\train'
# mat = Mat_voxelizer_EX_airplane(dir_path=airplanes, object_ratio=0.001)
# print(np.shape(mat))

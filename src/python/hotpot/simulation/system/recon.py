import numpy as np


def iteration(projection_data,matrix,image,effmap,num_iter):
    for i in range(num_iter):
        proj = matrix@image
        back_proj = np.transpose(matrix)@(projection_data/(proj+10**-10))
        image = image*back_proj/(effmap+10**-10)
    return image


# image = np.ones((3600,1))
effmap = np.transpose(sys_matrix)@np.ones((1279200,1))
effmap = effmap.reshape(10,90,90)
plt.imshow(effmap[5,:,:])


projection_data = np.load('/home/twj2417/tmp/bianhao.npy')
image = np.ones((90*90*10,1),np.float32)
output = iteration(projection_data,sys_matrix,image,effmap,5)
output = output.reshape(int(grid_image[2]),int(grid_image[1]),int(grid_image[0]))
plt.imshow(output[5,:,:])
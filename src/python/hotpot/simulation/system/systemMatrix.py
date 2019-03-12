import numpy as np
import math


config = {
    "scanner": {
        "ring": {
            "inner_radius": 99.0,
            "outer_radius": 119.0,
            "axial_length": 33.4,
            "nb_rings": 1,
            "nb_blocks_per_ring": 16,
            "gap": 0.0
        },
        "block": {
            "grid": [1,40,40],
            "size": [20.0,33.4,33.4]
        }
    },
    "image": {
            "grid": [320,320,10],
            "size": [150.0,150.0,33.4],
            "center": [0.0,0.0,0.0]
    }
}


def get_data_from_file(filename,grid_block,num_block):
    data = np.fromfile(filename,dtype=np.float32)
    data = data.reshape(-1,7)
    result = np.zeros((data.shape[0],5))
    result[:,0] = np.floor(data[:,1]/grid_block[1])*grid_block[1]*num_block+data[:,0]*grid_block[1]+np.mod(data[:,1],grid_block[1])
    result[:,1] = np.floor(data[:,3]/grid_block[1])*grid_block[1]*num_block+data[:,2]*grid_block[1]+np.mod(data[:,3],grid_block[1])
    result[:,2:5] = data[:,4:7]
    return result


def get_minor_edge(center_image,size_image):
    return center_image-size_image/2


def get_superior_edge(center_image,size_image):
    return center_image+size_image/2


def get_source_id(pos_source,size_image,grid_image,center_image):
    minor_edge = get_minor_edge(center_image,size_image)
    size_pixel = size_image/grid_image
    id_x = (pos_source[:,0]-minor_edge[0])//size_pixel[0]
    id_y = (pos_source[:,1]-minor_edge[1])//size_pixel[1]
    id_z = (pos_source[:,2]-minor_edge[2])//size_pixel[2]
    id_xy = np.hstack((id_x.reshape(id_x.size,1),id_y.reshape(id_y.size,1)))
    return np.hstack((id_xy,id_z.reshape(id_x.size,1)))


def get_row_id(id1,id2):
    return np.floor(id1*(id1-np.ones_like(id1))/2+id2)


def get_col_id(id_x,id_y,id_z,grid_image):
    return id_z*grid_image[0]*grid_image[1]+id_y*grid_image[0]+id_x


def rotate_crystal(crystal_id,num,total,nb_detectors_per_ring):
    ring_id = np.floor(crystal_id/nb_detectors_per_ring)
    crystal_per_ring = crystal_id - ring_id*nb_detectors_per_ring
    new_crystal_per_ring = np.mod(crystal_per_ring+int(num/total*nb_detectors_per_ring)*np.ones_like(crystal_per_ring),nb_detectors_per_ring)
    return ring_id*nb_detectors_per_ring+new_crystal_per_ring


def rotate_pos(pos,num,total):
    theta = 2*num/total*math.pi
    rotated_pos = np.array(pos)
    rotated_pos[:,0] = pos[:,0]*math.cos(theta)-pos[:,1]*math.sin(theta)
    rotated_pos[:,1] = pos[:,0]*math.sin(theta)+pos[:,1]*math.cos(theta)
    return rotated_pos


def rotate(data,num,total,nb_detectors_per_ring):
    result = np.array(data)
    result[:,0] = rotate_crystal(data[:,0],num,total,nb_detectors_per_ring)
    result[:,1] = rotate_crystal(data[:,1],num,total,nb_detectors_per_ring)
    result[:,2:5] = rotate_pos(data[:,2:5],num,total)
    return result


from scipy import sparse
def get_simu_matrix(filename,size_image,grid_image,center_image,grid_block,num_block):
    num_crystal_per_ring = grid_block[1]*num_block
    num_crystal = num_crystal_per_ring*grid_block[2]
    num_voxel = grid_image[0]*grid_image[1]*grid_image[2]
    data = get_data_from_file(filename,grid_block,num_block)

    sys_matrix = sparse.coo_matrix(([0],([0],[0])),shape=(int(num_crystal*(num_crystal-1)/2),num_voxel))
    for i in range(num_block):
        data = rotate(data,i,num_block,num_crystal_per_ring)

    source_id = get_source_id(data[:,2:],size_image,grid_image,center_image)
    index1 = np.where((source_id[:,0]>=0)
                      &(source_id[:,0]<grid_image[0])
                      &(source_id[:,1]>=0)
                      &(source_id[:,1]<grid_image[1])
                      &(source_id[:,2]>=0)
                      &(source_id[:,2]<grid_image[2]))[0]

    effective_data = data[index1,:2]
    effective_source_id = source_id[index1,:]
    row_id = get_row_id(effective_data[:,0],effective_data[:,1])
    col_id = get_col_id(effective_source_id[:,0],effective_source_id[:,1],effective_source_id[:,2],grid_image)
    value = np.ones_like(col_id)
    sys_matrix = sys_matrix + sparse.coo_matrix((value,(row_id,col_id)),shape=(int(num_crystal*(num_crystal-1)/2),num_voxel))
    return sys_matrix



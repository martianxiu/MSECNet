from __future__ import print_function
import os
import os.path
from os.path import exists, join
import sys
import math
import torch
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
from plyfile import PlyData, PlyElement
from util.transform import *  



# do NOT modify the returned points! kdtree uses a reference, not a copy of these points,
# so modifying the points would make the kdtree give incorrect results
def load_shape(point_filename, normals_filename, curv_filename, pidx_filename, pca_normals_filename):
    pts = np.load(point_filename+'.npy')

    if normals_filename != None:
        normals = np.load(normals_filename+'.npy')
    else:
        normals = None

    if curv_filename != None:
        curvatures = np.load(curv_filename+'.npy')
    else:
        curvatures = None

    if pidx_filename != None:
        patch_indices = np.load(pidx_filename+'.npy')
    else:
        patch_indices = None

    if pca_normals_filename != None:
        k = [18, 112, 450]
        _, normals_k0 = read_ply_with_normal(pca_normals_filename + f'_k{k[0]}.ply')
        _, normals_k1 = read_ply_with_normal(pca_normals_filename + f'_k{k[1]}.ply')
        _, normals_k2 = read_ply_with_normal(pca_normals_filename + f'_k{k[2]}.ply')
        ms_pca_normals = np.concatenate([normals_k0, normals_k1, normals_k2], axis=1) # (n, 9)
    else:
        ms_pca_normals = None

    sys.setrecursionlimit(int(max(1000, round(pts.shape[0]/10)))) # otherwise KDTree construction may run out of recursions
    kdtree = spatial.cKDTree(pts, 10)

    return Shape(pts=pts, kdtree=kdtree, normals=normals, curv=curvatures, pidx=patch_indices, ms_pca_normals=ms_pca_normals)

class SequentialPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = None

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + self.data_source.shape_patch_count[shape_ind]

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


class SequentialShapeRandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, sequential_shapes=False, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.sequential_shapes = sequential_shapes
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None
        self.shape_patch_inds = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        # global point index offset for each shape
        shape_patch_offset = list(np.cumsum(self.data_source.shape_patch_count))
        shape_patch_offset.insert(0, 0)
        shape_patch_offset.pop()

        shape_inds = range(len(self.data_source.shape_names))

        if not self.sequential_shapes:
            shape_inds = self.rng.permutation(shape_inds)

        # return a permutation of the points in the dataset where all points in the same shape are adjacent (for performance reasons):
        # first permute shapes, then concatenate a list of permuted points in each shape
        self.shape_patch_inds = [[]]*len(self.data_source.shape_names)
        point_permutation = []
        for shape_ind in shape_inds:
            start = shape_patch_offset[shape_ind]
            end = shape_patch_offset[shape_ind]+self.data_source.shape_patch_count[shape_ind]

            global_patch_inds = self.rng.choice(range(start, end), size=min(self.patches_per_shape, end-start), replace=False)
            point_permutation.extend(global_patch_inds)

            # save indices of shape point subset
            self.shape_patch_inds[shape_ind] = global_patch_inds - start

        return iter(point_permutation)

    def __len__(self):
        return self.total_patch_count

class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


import torch.distributed as dist
class RandomPointcloudPatchSamplerDistributed(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False, num_replicas=None, rank=None, shuffle=True, drop_last=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)) 
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.identical_epochs = identical_epochs
        self.total_patch_count = None
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.epoch = 0

        self.total_patch_count = 0 # how many patches per iteration. shape_patch_count is the number of points because by default every point is a patch center.   
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

        self.num_samples = math.ceil(self.total_patch_count / self.num_replicas) # divided by gpu 
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        else:
            self.rng = np.random.RandomState(self.seed)

    def __iter__(self):
        # Each point in shapes is the potential patch center. Denoted by sum(self.data_source.shape_patch_count)
        # From all points extract #shape x self.patches_per_shape as training.
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(int(self.seed + self.epoch))
            indices = torch.randperm(sum(self.data_source.shape_patch_count), generator=g).tolist() 
        else:
            indices = list(range(sum(self.data_source.shape_patch_count)))
        # indice: summed number of all points in shapes   

        # subsample self.total_size  from all patch centers  
        indices = indices[:self.total_size] # here indices are already shuffled 
        assert len(indices) == self.total_size

        # subsample for each GPU 
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    # def __iter__(self):
    #     return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False)) # randomly pick patches 

    def __len__(self):
        # return self.total_patch_count
        return self.num_samples # num pathces divided by replicas 

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class Shape():
    def __init__(self, pts, kdtree, normals=None, curv=None, pidx=None, ms_pca_normals=None):
        self.pts = pts
        self.kdtree = kdtree
        self.normals = normals
        self.curv = curv
        self.pidx = pidx # patch center points indices (None means all points are potential patch centers)
        self.ms_pca_normals = ms_pca_normals


class Cache():
    def __init__(self, capacity, loader, loadfunc):
        self.elements = {}
        self.used_at = {}
        self.capacity = capacity
        self.loader = loader
        self.loadfunc = loadfunc
        self.counter = 0

    def get(self, element_id):
        if element_id not in self.elements:
            # cache miss

            # if at capacity, throw out least recently used item
            if len(self.elements) >= self.capacity:
                remove_id = min(self.used_at, key=self.used_at.get)
                del self.elements[remove_id]
                del self.used_at[remove_id]

            # load element
            self.elements[element_id] = self.loadfunc(self.loader, element_id) # load a shape

        self.used_at[element_id] = self.counter
        self.counter += 1

        return self.elements[element_id]


class PointcloudPatchDataset(data.Dataset):

    # patch radius as fraction of the bounding box diagonal of a shape
    def __init__(self, config, root, shape_list_filename, patch_radius, points_per_patch, patch_features,
                 seed=None, identical_epochs=False, use_pca=True, center='point', point_tuple=1, cache_capacity=1, point_count_std=0.0, sparse_patches=False, sampling='radius', transform=None, split=None, pp_normal=False):

        # initialize parameters
        self.root = root
        self.shape_list_filename = shape_list_filename # shape in the split 
        self.patch_features = patch_features
        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.identical_epochs = identical_epochs
        self.use_pca = use_pca
        self.sparse_patches = sparse_patches
        self.center = center
        self.point_tuple = point_tuple
        self.point_count_std = point_count_std
        self.seed = seed
        self.sampling = sampling
        self.transform = transform 
        self.split = split
        self.pp_normal = pp_normal
        self.ms_pca_normals = config.ms_pca_normals

        self.include_normals = False
        self.include_curvatures = False
        for pfeat in self.patch_features:
            if pfeat == 'normal':
                self.include_normals = True
            elif pfeat == 'max_curvature' or pfeat == 'min_curvature':
                self.include_curvatures = True
            else:
                raise ValueError('Unknown patch feature: %s' % (pfeat))

        # self.loaded_shape = None
        self.load_iteration = 0
        self.shape_cache = Cache(cache_capacity, self, PointcloudPatchDataset.load_shape_by_index)

        # get all shape names in the dataset
        self.shape_names = []
        with open(os.path.join(root, self.shape_list_filename)) as f:
            self.shape_names = f.readlines()
        self.shape_names = [x.strip() for x in self.shape_names]
        self.shape_names = list(filter(None, self.shape_names))

        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        # get basic information for each shape in the dataset
        self.shape_patch_count = []
        self.patch_radius_absolute = []
        for shape_ind, shape_name in enumerate(self.shape_names):
            print(f'[{self.split}] getting information for shape {shape_name}' )

            # load from text file and save in more efficient numpy format
            point_filename = os.path.join(self.root, shape_name+'.xyz')
            if not exists(point_filename+'.npy'):
                pts = np.loadtxt(point_filename).astype('float32')
                np.save(point_filename+'.npy', pts)
                # print(f'.xyz --> .npy, {point_filename}')

            if self.include_normals:
                normals_filename = os.path.join(self.root, shape_name+'.normals')
                if not exists(normals_filename+'.npy'):
                    normals = np.loadtxt(normals_filename).astype('float32')
                    np.save(normals_filename+'.npy', normals)
            else:
                normals_filename = None

            if self.include_curvatures:
                curv_filename = os.path.join(self.root, shape_name+'.curv')
                if not exists(curv_filename+'.npy'):
                    curvatures = np.loadtxt(curv_filename).astype('float32')
                    np.save(curv_filename+'.npy', curvatures)
            else:
                curv_filename = None

            if self.sparse_patches:
                pidx_filename = os.path.join(self.root, shape_name+'.pidx')
                if not exists(pidx_filename+'.npy'):
                    patch_indices = np.loadtxt(pidx_filename).astype('int')
                    np.save(pidx_filename+'.npy', patch_indices)
            else:
                pidx_filename = None

            shape = self.shape_cache.get(shape_ind)

            if shape.pidx is None:
                self.shape_patch_count.append(shape.pts.shape[0])
            else:
                self.shape_patch_count.append(len(shape.pidx))

            bbdiag = float(np.linalg.norm(shape.pts.max(0) - shape.pts.min(0), 2)) # bounding box
            self.patch_radius_absolute.append([bbdiag * rad for rad in self.patch_radius]) # radis proportional to the bounding box 

    # returns a patch centered at the point with the given global index
    # and the ground truth normal the the patch center
    def __getitem__(self, index):

        # find shape that contains the point with given global index
        shape_ind, patch_ind = self.shape_index(index)

        shape = self.shape_cache.get(shape_ind)
        if shape.pidx is None:
            center_point_ind = patch_ind
        else:
            center_point_ind = shape.pidx[patch_ind]

        patch_pts = np.zeros((self.points_per_patch*len(self.patch_radius_absolute[shape_ind]), 3)) # container.
        patch_pts_valid = []
        scale_ind_range = np.zeros([len(self.patch_radius_absolute[shape_ind]), 2], dtype='int')
        for s, rad in enumerate(self.patch_radius_absolute[shape_ind]):
            if self.sampling == 'radius':
                patch_point_inds = np.array(shape.kdtree.query_ball_point(shape.pts[center_point_ind, :], rad)) # ball query
            elif self.sampling == 'knn':
                patch_point_inds = np.array(shape.kdtree.query(shape.pts[center_point_ind, :], self.points_per_patch))[1].astype(int) # k-neighbors, in this way the first point is the center point. 

            # optionally always pick the same points for a given patch index (mainly for debugging)
            if self.identical_epochs:
                self.rng.seed((self.seed + index) % (2**32))

            point_count = min(self.points_per_patch, len(patch_point_inds))

            # randomly decrease the number of points to get patches with different point densities
            if self.point_count_std > 0:
                point_count = max(5, round(point_count * self.rng.uniform(1.0-self.point_count_std*2)))
                point_count = min(point_count, len(patch_point_inds))

            # if there are too many neighbors, pick a random subset. (with k-nn, there will never happen)
            if point_count < len(patch_point_inds):
                patch_point_inds = patch_point_inds[self.rng.choice(len(patch_point_inds), point_count, replace=False)]

            start = s*self.points_per_patch # start from s * num_point, if #points is smaller than num_point, zero padded (0,0,0). 
            end = start+point_count
            scale_ind_range[s, :] = [start, end]

            patch_pts_valid += list(range(start, end))

            patch_pts[start:end, :] = shape.pts[patch_point_inds, :]

            # center patch (central point at origin - but avoid changing padded zeros)
            if self.center == 'mean':
                patch_pts[start:end, :] = patch_pts[start:end, :] - patch_pts[start:end, :].mean(0)
            elif self.center == 'point':
                # patch_pts[start:end, :] = patch_pts[start:end, :] - torch.from_numpy(shape.pts[center_point_ind, :])
                patch_pts[start:end, :] = patch_pts[start:end, :] - shape.pts[center_point_ind, :]
            elif self.center == 'none':
                pass # no centering
            else:
                raise ValueError('Unknown patch centering option: %s' % (self.center))

            # normalize size of patch (scale with 1 / patch radius)
            if self.sampling == 'radius':
                patch_pts[start:end, :] = patch_pts[start:end, :] / rad
            elif self.sampling == 'knn':
                patch_pts[start:end, :] = patch_pts[start:end, :] / (np.max(np.sum(patch_pts[start:end, :] ** 2, axis=1)**0.5, axis=0) + 1e-10) 


        if self.include_normals:
            if self.pp_normal:
                patch_normal = shape.normals[patch_point_inds, :] # take all normals of the patch 
                if self.ms_pca_normals:
                    patch_ms_pca_normals = shape.ms_pca_normals[patch_point_inds, :] # 
            # patch_normal = torch.from_numpy(shape.normals[center_point_ind, :])
            else:
                patch_normal = shape.normals[center_point_ind, :]
                if self.ms_pca_normals:
                    patch_ms_pca_normals = shape.ms_pca_normals[center_point_ind, :] # 

        if self.include_curvatures:
            # patch_curv = torch.from_numpy(shape.curv[center_point_ind, :])
            patch_curv = shape.curv[center_point_ind, :]
            # scale curvature to match the scaled vertices (curvature*s matches position/s):
            patch_curv = patch_curv * self.patch_radius_absolute[shape_ind][0]


        # get point tuples from the current patch
        if self.point_tuple > 1:
            # patch_tuples = torch.zeros(self.points_per_patch*len(self.patch_radius_absolute[shape_ind]), 3*self.point_tuple, dtype=torch.float)
            patch_tuples = np.zeros(self.points_per_patch*len(self.patch_radius_absolute[shape_ind]), 3*self.point_tuple)
            for s, rad in enumerate(self.patch_radius_absolute[shape_ind]):
                start = scale_ind_range[s, 0]
                end = scale_ind_range[s, 1]
                point_count = end - start

                tuple_count = point_count**self.point_tuple

                # get linear indices of the tuples
                if tuple_count > self.points_per_patch:
                    patch_tuple_inds = self.rng.choice(tuple_count, self.points_per_patch, replace=False)
                    tuple_count = self.points_per_patch
                else:
                    patch_tuple_inds = np.arange(tuple_count)

                # linear tuple index to index for each tuple element
                patch_tuple_inds = np.unravel_index(patch_tuple_inds, (point_count,)*self.point_tuple)

                for t in range(self.point_tuple):
                    patch_tuples[start:start+tuple_count, t*3:(t+1)*3] = patch_pts[start+patch_tuple_inds[t], :]


            patch_pts = patch_tuples

        patch_feats = ()
        for pfeat in self.patch_features:
            if pfeat == 'normal':
                patch_feats = patch_feats + (patch_normal.reshape(-1, 3),)
            elif pfeat == 'max_curvature':
                patch_feats = patch_feats + (patch_curv[0:1],)
            elif pfeat == 'min_curvature':
                patch_feats = patch_feats + (patch_curv[1:2],)
            else:
                raise ValueError('Unknown patch feature: %s' % (pfeat))


        if self.ms_pca_normals:
            patch_feats = patch_feats + (patch_ms_pca_normals.reshape(-1, 9), )
        patch_feats = np.concatenate(patch_feats, 1)
        coord = patch_pts # (n, 3)
        feat = patch_feats # (n, n_feat)
        label = np.array([0])

        
        # pca to transfrom into a canonical form 
        if self.use_pca:
            pca = PCARotate(return_trans=True)
            coord, feat, label, trans = pca(coord, feat, label)
        else:
            trans = np.eye((3,3)).astype(np.float32) 

        # augmentation
        if self.transform is not None:
            coord, feat, label = self.transform(coord, feat, label)

        return torch.FloatTensor(np.array(coord)), torch.FloatTensor(np.array(feat)), torch.LongTensor(np.array([label])), torch.FloatTensor(np.array(trans))


    def __len__(self):
        return sum(self.shape_patch_count)


    # translate global (dataset-wide) point index to shape index & local (shape-wide) point index
    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind

    # load shape from a given shape index
    def load_shape_by_index(self, shape_ind):
        point_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.xyz')
        normals_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.normals') if self.include_normals else None
        curv_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.curv') if self.include_curvatures else None
        pidx_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.pidx') if self.sparse_patches else None
        pca_normals_filename = os.path.join(self.root, 'pca_normals', self.shape_names[shape_ind]) if self.ms_pca_normals else None
        return load_shape(point_filename, normals_filename, curv_filename, pidx_filename, pca_normals_filename)

def read_ply_with_normal(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    x, y, z = pc['x'], pc['y'], pc['z']
    nx, ny, nz = pc['nx'], pc['ny'], pc['nz']
    xyz = np.vstack([x, y, z]).T
    normals = np.vstack([nx, ny, nz]).T
    return xyz, normals
# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Transform points to voxels using sparse conv library.

This version keeps the original OpenCOOD behavior and adds CoopDiff's
``preprocess_paint`` path. It also avoids importing ``cumm`` at module import
time so spconv v1 environments will not fail before the v1 branch is selected.
"""
import sys

import numpy as np
import torch

from opencood.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor


class SpVoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(SpVoxelPreprocessor, self).__init__(preprocess_params, train)

        self.spconv = 1
        self._tv = None
        VoxelGenerator = None
        last_error = None

        # spconv v1.x
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv = 1
        except Exception as err:
            last_error = err

        # Some spconv v1 builds expose VoxelGenerator instead of VoxelGeneratorV2.
        if VoxelGenerator is None:
            try:
                from spconv.utils import VoxelGenerator as VoxelGenerator
                self.spconv = 1
            except Exception as err:
                last_error = err

        # spconv v2.x used by many OpenCOOD environments.
        if VoxelGenerator is None:
            try:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                from cumm import tensorview as tv
                self._tv = tv
                self.spconv = 2
            except Exception as err:
                last_error = err

        # Newer spconv v2 package layout.
        if VoxelGenerator is None:
            try:
                from spconv.pytorch.utils import PointToVoxel as VoxelGenerator
                from cumm import tensorview as tv
                self._tv = tv
                self.spconv = 2
            except Exception as err:
                last_error = err

        if VoxelGenerator is None:
            raise ImportError(
                "spconv is required by SpVoxelPreprocessor but cannot be imported. "
                "Install the spconv build matching your CUDA/PyTorch version, e.g. "
                "`pip install spconv-cu113` for CUDA 11.3, or `pip install spconv-cu114` "
                "for CUDA 11.4. If your OpenCOOD environment is CUDA 11.1, use "
                "`pip install spconv-cu111`."
            ) from last_error

        self.lidar_range = self.params['cav_lidar_range']
        self.voxel_size = self.params['args']['voxel_size']
        self.max_points_per_voxel = self.params['args']['max_points_per_voxel']

        if train:
            self.max_voxels = self.params['args']['max_voxel_train']
        else:
            self.max_voxels = self.params['args']['max_voxel_test']

        grid_size = (np.array(self.lidar_range[3:6]) -
                     np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        # use sparse conv library to generate voxel
        if self.spconv == 1:
            self.voxel_generator = VoxelGenerator(
                voxel_size=self.voxel_size,
                point_cloud_range=self.lidar_range,
                max_num_points=self.max_points_per_voxel,
                max_voxels=self.max_voxels
            )
            # spconv v1 generator infers feature dimension from the input array.
            self.voxel_generator_paint = self.voxel_generator
        else:
            self.voxel_generator = VoxelGenerator(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.lidar_range,
                max_num_points_per_voxel=self.max_points_per_voxel,
                num_point_features=4,
                max_num_voxels=self.max_voxels
            )
            # CoopDiff uses painted lidar with 5 point features.
            self.voxel_generator_paint = VoxelGenerator(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.lidar_range,
                max_num_points_per_voxel=self.max_points_per_voxel,
                num_point_features=5,
                max_num_voxels=self.max_voxels
            )

    def _point_to_voxel(self, generator, pcd_np):
        """Run the selected spconv voxel generator and normalize output."""
        pcd_np = np.ascontiguousarray(pcd_np)

        if self.spconv == 1:
            voxel_output = generator.generate(pcd_np)
        else:
            pcd_tv = self._tv.from_numpy(pcd_np)
            voxel_output = generator.point_to_voxel(pcd_tv)

        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], \
                voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if self.spconv == 2:
            voxels = voxels.numpy()
            coordinates = coordinates.numpy()
            num_points = num_points.numpy()

        return {
            'voxel_features': voxels,
            'voxel_coords': coordinates,
            'voxel_num_points': num_points
        }

    def preprocess(self, pcd_np):
        return self._point_to_voxel(self.voxel_generator, pcd_np)

    def preprocess_paint(self, pcd_np):
        """
        CoopDiff painted-lidar preprocessing.

        Expected input shape is [N, 5]. If [N, 4] is passed accidentally,
        append a zero-valued paint channel instead of crashing immediately.
        """
        if pcd_np.shape[1] == 4:
            pad = np.zeros((pcd_np.shape[0], 1), dtype=pcd_np.dtype)
            pcd_np = np.concatenate([pcd_np, pad], axis=1)
        return self._point_to_voxel(self.voxel_generator_paint, pcd_np)

    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            List or dictionary.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """

        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit('Batch has too be a list or a dictionarn')

    @staticmethod
    def collate_batch_list(batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represent a single frame.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = []
        voxel_num_points = []
        voxel_coords = []

        for i in range(len(batch)):
            voxel_features.append(batch[i]['voxel_features'])
            voxel_num_points.append(batch[i]['voxel_num_points'])
            coords = batch[i]['voxel_coords']
            voxel_coords.append(
                np.pad(coords, ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))

        voxel_num_points = torch.from_numpy(np.concatenate(voxel_num_points))
        voxel_features = torch.from_numpy(np.concatenate(voxel_features))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points}

    @staticmethod
    def collate_batch_dict(batch: dict):
        """
        Collate batch if the batch is a dictionary,
        eg: {'voxel_features': [feature1, feature2...., feature n]}

        Parameters
        ----------
        batch : dict

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = \
            torch.from_numpy(np.concatenate(batch['voxel_features']))
        voxel_num_points = \
            torch.from_numpy(np.concatenate(batch['voxel_num_points']))
        coords = batch['voxel_coords']
        voxel_coords = []

        for i in range(len(coords)):
            voxel_coords.append(
                np.pad(coords[i], ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points}

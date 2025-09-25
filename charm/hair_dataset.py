import json
import os

import numpy as np  
import open3d as o3d
import torch
from torch.utils.data import Dataset

from .utils.logger import print_log


def create_dataset(cfg_dataset):
    kwargs = cfg_dataset
    name = kwargs.pop('name')
    dataset = get_dataset(name)(**kwargs)
    print_log(f"Dataset '{name}' init: kwargs={kwargs}, len={len(dataset)}")
    return dataset

def get_dataset(name):
    return {
        'dev': HairDataset,
    }[name]


class HairDataset(Dataset): 
    def __init__(self,
        file_list='',
        file_dir='',
        pc_dir='',
        max_length=8192,
        range_translation=[-0.5, 0.5],
        range_width=[0.0, 0.1],
        range_thickness=[0.0, 0.1],
        use_pc=False,
        pc_format='pc',
        return_file_path=False
    ):
        assert isinstance(file_list, str) and file_list.endswith('.json') or pc_dir != ''

        if file_list != '':
            with open(file_list, 'r') as f:
                data = json.load(f)

            self.data_filename = [os.path.join(file_dir, item, 'hair.json') for item in data]
        elif pc_dir != '':
            self.data_filename = sorted([os.path.join(pc_dir, item) for item in os.listdir(pc_dir)])
        else:
            raise ValueError('file_list and pc_dir cannot be empty at the same time')

        self.max_length = max_length
        self.range_translation = range_translation
        self.range_width = range_width
        self.range_thickness = range_thickness
        self.use_pc = use_pc
        self.pc_format = pc_format
        self.return_file_path = return_file_path

    def __len__(self):
        return len(self.data_filename)

    def __getitem__(self, idx):
        if self.data_filename[idx].endswith('.json'):
            json_file = self.data_filename[idx]
            with open(json_file, 'r') as f:
                model_json = json.load(f)
            _, model_data = self.parse_json(model_json, enable_check=False)
        else:
            model_data = {}

        if self.use_pc:
            if self.data_filename[idx].endswith('.json'):
                pc_file = json_file.replace('hair.json', 'hair_reform.ply')
            else:
                pc_file = self.data_filename[idx]
            pc = o3d.io.read_point_cloud(pc_file)

        if self.use_pc:
            points = torch.from_numpy(np.asarray(pc.points)).float()
            colors = torch.from_numpy(np.asarray(pc.colors)).float()
            normals = torch.from_numpy(np.asarray(pc.normals)).float()

            # normalize points to [-1, 1]
            points *= 2.

            if self.pc_format == 'pc':
                model_data['pc'] = torch.concatenate([points, colors], dim=-1).T
            elif self.pc_format == 'pn':
                model_data['pc'] = torch.concatenate([points, normals], dim=-1)
            elif self.pc_format == 'pcn':
                model_data['pc'] = torch.concatenate([points, colors, normals], dim=-1)
            else:
                raise ValueError(f'invalid pc_format: {self.pc_format}')

        if self.return_file_path:
            model_data['file_path'] = json_file

        return model_data

    def parse_json(self, model, enable_check=True):
        if enable_check and len(model['group']) > self.max_length:
            return False, None

        hair_list = []
        mean_translation_list = []

        max_ys = []
        for block in model:
            seq = block['seq']
            vertices = np.array([s[:3] for s in seq])
            max_y = np.max(vertices[:, 1])
            max_ys.append(max_y)
        max_y = np.percentile(max_ys, 95)
        anchor = np.array([0, max_y, 0])

        for block in model:
            translation_list = []
            width_list = []
            thickness_list = []
            for segment in block['seq']:
                translation = segment[:3]
                width = segment[3]
                thickness = segment[4]
                translation_list.append(translation)
                width_list.append(width)
                thickness_list.append(thickness)

            # check value range
            translation_array = np.array(translation_list)
            width_array = np.array(width_list)
            thickness_array = np.array(thickness_list)
            if enable_check and (
                not check_valid_range(translation_array, self.range_translation) or \
                not check_valid_range(width_array, self.range_width) or \
                not check_valid_range(thickness_array, self.range_thickness)
            ):
                return False, None

            # check direction
            dis0 = np.linalg.norm(translation_array[0] - anchor)
            dis1 = np.linalg.norm(translation_array[-1] - anchor)
            if dis0 > dis1:
                translation_array = translation_array[::-1].copy()
                width_array = width_array[::-1].copy()
                thickness_array = thickness_array[::-1].copy()

            hair_list.append({
                'translation': torch.from_numpy(translation_array).float(),
                'width': torch.from_numpy(width_array).float(),
                'thickness': torch.from_numpy(thickness_array).float(),
            })
            mean_translation_list.append(np.mean(translation_array, axis=0))
        
        mean_translation_list = np.array(mean_translation_list)
        arctan2 = np.arctan2(mean_translation_list[:, 0], mean_translation_list[:, 2])
        order = np.argsort(arctan2)

        hair_list = [hair_list[i] for i in order]

        return True, {
            'hair': hair_list,
        }


def check_valid_range(data, value_range):
    lo, hi = value_range
    assert hi > lo
    return np.logical_and(data >= lo, hi <= hi).all()


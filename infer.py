import argparse
import json
import os

from tqdm import tqdm
import torch

from charm.hair_dataset import create_dataset
from charm.hair_transformer_trainer import create_model as create_AR_model
from charm.utils import load_yaml_no_default, torch_to

from hair_utils.tmp2mesh import hairtemplate2mesh


def write_json(hairs, out_path):
    new_block = {}
    new_block["shape"] = "square"
    new_block['seq'] = []

    new_group = []
    for translation, width, thickness, mos in zip(
        hairs['translation'].squeeze().cpu().numpy(),
        hairs['width'].squeeze().cpu().numpy(),
        hairs['thickness'].squeeze().cpu().numpy(),
        hairs['mos_codes'].squeeze().cpu().numpy()
    ):
        if width == -1:
            break
        if mos == True:
            new_group.append(new_block)
            new_block = {}
            new_block["shape"] = "square"
            new_block['seq'] = []
            continue
        new_block['seq'].append([float(translation[0]), float(translation[1]), float(translation[2]), float(width), float(thickness)])
    new_group.append(new_block)

    with open(out_path, 'w') as json_file:
        json.dump(new_group, json_file, indent=4)

    meshes = hairtemplate2mesh(str(out_path))
    meshes.export(str(out_path).replace(".json", ".glb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./configs/infer.yml', help='Config file name')
    parser.add_argument('-ck', '--AR_ckpt', type=str, default='./ckpt/charm.safetensors')
    parser.add_argument('-i', '--input_dir', type=str, default='./test_cases/pc')
    parser.add_argument('-o', '--output_dir', type=str, default='./results')
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()

    cfg = load_yaml_no_default(args.config)
    AR_checkpoint = torch.load(args.AR_ckpt)
    infer_folder = args.output_dir
    
    os.makedirs(infer_folder, exist_ok=True)
    json_result_folder = os.path.join(infer_folder, 'JsonResults')
    os.makedirs(json_result_folder, exist_ok=True)

    dataset = create_dataset(cfg['dataset'])

    transformer = create_AR_model(cfg['model'])
    transformer.load_state_dict(AR_checkpoint, strict=False)

    for item_i, item in tqdm(enumerate(dataset)):
        pc = item.pop('pc')

        item_filename = dataset.data_filename[item_i]
        print(item_filename)
        if torch.cuda.is_available():
            pc = pc.cuda()
            item = torch_to(item, torch.device('cuda'))
            transformer = transformer.cuda()

        recon_primitives, mask = transformer.generate(pc=pc.unsqueeze(0), temperature=args.temperature)

        out_path = os.path.join(json_result_folder, os.path.basename(item_filename).replace('.ply', '.json'))
        write_json(recon_primitives, out_path)

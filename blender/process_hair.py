import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from hair_utils.mesh2tmp import mesh2hairtemplate
from hair_utils.tmp2mesh import hairtemplate2mesh
import glob
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Process hair GLB files in a directory.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the root directory containing hair.glb files"
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="./blender/process_hair.log",
        help="Path to the log file."
    )
    args = parser.parse_args()

    search_pattern = os.path.join(args.input_dir, "*", "*", "hair.glb")
    ls = glob.glob(search_pattern)
    log_path = args.log_path

    for mesh_path in tqdm(ls):
        tmp_path = mesh_path.replace("hair.glb", "hair.json")
        if os.path.exists(tmp_path.replace("hair.json", "hair_reform.glb")):
            continue

        try:
            tmp, len_hairs = mesh2hairtemplate(mesh_path)
            with open(tmp_path, "w") as f:
                json.dump(tmp, f, indent=4)

            meshes = hairtemplate2mesh(tmp_path)
            meshes.export(mesh_path.replace("hair.glb", "hair_reform.glb"))
            with open(log_path, "a") as f:
                f.write(f"Success:\t{len_hairs}\t{mesh_path}\n")
        except Exception as e:
            print(f"Error processing {mesh_path}: {e}")
            with open(log_path, "a") as f:
                f.write(f"Error:\t{e}\t{mesh_path}\n")
            continue

if __name__ == "__main__":
    main()

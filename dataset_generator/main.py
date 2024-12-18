import os
from utils import *
from argparse import ArgumentParser

def main(args):
    data_root = args.data_root
    num_samples_str = ' '.join(map(str, args.num_samples))
    output_dir = args.output_dir

    for scene in os.listdir(data_root):
        scene_path = os.path.join(data_root, scene)
        output_path = os.path.join(output_dir, scene)
        cmd = f"python dataset_generator/train_render.py --scene_path {scene_path} --output_path {output_path} --num_samples {num_samples_str} --use_lr"
        os.system(cmd)



parser = ArgumentParser(description="Data generator")
parser.add_argument("--data_root", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--num_samples", nargs="+", type=int, default=[3, 6,9])

args = parser.parse_args()

main(args)




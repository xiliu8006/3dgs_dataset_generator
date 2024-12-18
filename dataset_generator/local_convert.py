from argparse import ArgumentParser
from utils import bin_to_txt
import os

parser = ArgumentParser(description="Locally convert the colmap bin to txt")
parser.add_argument("--data_root", type=str)

args = parser.parse_args()

data_root = args.data_root

for scene in os.listdir(data_root):
    scene_path = os.path.join(data_root, scene)
    bin_to_txt(scene_path)
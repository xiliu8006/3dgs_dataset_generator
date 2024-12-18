from argparse import ArgumentParser
import os
from dataset_generator.utils import train_render_gss

parser = ArgumentParser(description="Data generator for each scene")
parser.add_argument("--scene_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--num_samples", nargs="+", type=int, default=[3, 6, 9, 24])
parser.add_argument("--use_lr",action="store_true")
parser.add_argument("--port", type=int)

args = parser.parse_args()

train_render_gss(args.scene_path, args.output_path, args.num_samples, args.use_lr, args.port)

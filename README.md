# 3DGS-Enhancer Dataset Generation

**Authors:** Xi Liu\*, Chaoyi Zhou\*, Siyu Huang (* indicates equal contribution)

---

This repository contains the official implementation from the authors of the paper:  
"3DGS-Enhancer: Enhancing Unbounded 3D Gaussian Splatting with View-Consistent 2D Diffusion Priors."

We provide the resources here to facilitate the generation of paired datasets. Our dataset is built upon the [DL3DV-10K](https://dl3dv-10k.github.io/DL3DV-10K/) dataset, and this codebase is derived from the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) project. We thank the developers of these works for their significant contributions.

---

## How to Generate the Dataset

We provide a shell script to help generate the dataset. Please ensure that you modify the script to suit your server environment.

### Command:
```bash
bash train_multi_nodes.sh
```

### Setting:
This dataset will produce paired results for 3, 6, 9, or 24 views. You can modify the --num_samples argument in the train_render.py script to specify the desired number of views.
The rendering results can be matched with the ground truth images using their filenames. Reference view images will include the suffix _ref in their filenames.

### acknowledgement
If you find this codebase helpful for your research, please consider citing the three papers mentioned:

import os
import logging
import shutil
import  numpy as np
import collections
from .colmap_utils import read_model, write_model, BaseImage
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import json
from scipy.spatial.transform import Slerp
from PIL import Image
colmap_command = "colmap"


def bin_to_txt(scene_path):
    print(f"Convering scene {scene_path} ... ")
    sparse_path = os.path.join(scene_path, "sparse/0")
    input_path =  sparse_path
    output_path = sparse_path
    model_convert_cmd = (colmap_command + " model_converter \
                                                --input_path " + input_path + " \
                                                --output_path " + output_path + "\
                                                --output_type TXT")
    exit_code = os.system(model_convert_cmd)
    if exit_code != 0:
        logging.error(f"{scene_path} convertion failed with code {exit_code}. Exiting.")
        exit(exit_code)


# def train_lr(scene_path, output_path, num_sample, port):
#     output_path = os.path.join(output_path, 'lr', str(num_sample))

#     common_args = " --quiet --eval --test_iterations -1 "
#     train_cmd = "python train.py -s " + scene_path + " -i images_4 -m " + output_path +" --port "+ str(port) + common_args +f" --n_sparse {num_sample} --rand_init"
#     print(train_cmd)
#     if os.path.exists(os.path.join(output_path, "point_cloud/iteration_30000/point_cloud.ply")):
#         print(f"The inputs {num_sample} for {scene_path} has been created. Skip train")
#         return output_path
#     else:
#         os.system(train_cmd)
#     return output_path

def select_lr_cams(scene_path, num_samples):
    cameras, images = read_model(path=os.path.join(scene_path, "colmap/sparse/0"), ext=".bin")
    train_cam_infos = [c for idx, c in images.items()]
    train_cam_infos_sorted = sorted(train_cam_infos.copy(), key=lambda x: x.name)

    frame_gap = 50
    if num_samples == 3 and num_samples * 50 < len(train_cam_infos_sorted):
        frame_gap = 50
    if ((frame_gap * num_samples) > len(train_cam_infos_sorted)):
        frame_gap = 25
    
    # last level frame gap
    if ((frame_gap * num_samples) > len(train_cam_infos_sorted)):
        frame_gap = int(len(train_cam_infos_sorted) / num_samples)
    
    lr_train_cam_infos = []
    for i in range(num_samples):
        lr_train_cam_infos.append(train_cam_infos_sorted[i * frame_gap])
    return cameras, {image.id: image for image in lr_train_cam_infos}

def uniform_sample_with_fixed_count(lst, count):
    if count <= 2:
        return [lst[0], lst[-1]] if len(lst) > 1 else [lst[0]]

    sampled_list = [lst[0]]  # Start with the first element
    interval = (len(lst) - 1) / (count - 1)  # Calculate interval for count-1 divisions

    for i in range(1, count - 1):
        index = int(round(i * interval))  # Calculate index at each interval step
        sampled_list.append(lst[index])

    sampled_list.append(lst[-1])  # Ensure the last element is added
    return sampled_list

def select_eval_cams(scene_path, num_samples):
    cameras, images = read_model(path=os.path.join(scene_path, "colmap/sparse/0"), ext=".bin")
    train_cam_infos = [c for idx, c in images.items()]
    train_cam_infos_sorted = sorted(train_cam_infos.copy(), key=lambda x: x.name)
    lr_train_cam_infos = uniform_sample_with_fixed_count(train_cam_infos_sorted, num_samples)
    return cameras, {image.id: image for image in lr_train_cam_infos}

def check_valid(scene_path):
    image_path = os.path.join(scene_path, 'images_4')
    frames = os.listdir(image_path)
    min_width, min_height = 960, 540  # 960p resolution is generally 1280x720
    

    for frame in frames:
        frame_path = os.path.join(image_path, frame)
        with Image.open(frame_path) as img:
            width, height = img.size

            # Check if the resolution is lower than 960p
            if width != min_width or height != min_height:
                return False

    print("All frames are at least 960p.")
    return True


def train_lr(scene_path, output_path, num_sample, port, eval_mode=False):
    output_path = os.path.join(output_path, 'lr', str(num_sample))
    train_cam_path = os.path.join(output_path, "sparse/0")
    os.makedirs(train_cam_path, exist_ok=True)
    if eval_mode:
        cameras, selected_images = select_eval_cams(scene_path, num_sample)
    else:
        cameras, selected_images = select_lr_cams(scene_path, num_sample)
    # print("select_lr_cams len: ", len(selected_images), num_sample, train_cam_path)
    write_model(cameras, selected_images, train_cam_path, ext=".txt")
    
    common_args = " --quiet --test_iterations -1 "
    train_cmd = "python train.py -s " + scene_path + " -i images_4 -m " + output_path +" --port "+ str(port) + common_args +f" --n_sparse {num_sample} --train_lr --rand_init"
    print(train_cmd)
    if os.path.exists(os.path.join(output_path, "point_cloud/iteration_30000/point_cloud.ply")):
        print(f"The inputs {num_sample} for {scene_path} has been created. Skip train")
        return output_path
    else:
        os.system(train_cmd)
    return output_path

def train_hr(scene_path, output_path, port):
    common_args = " --quiet --test_iterations -1 "
    output_path = os.path.join(output_path, "hr")
    train_cmd = "python train.py -s " + scene_path + " -i images_4 -m" + output_path  +" --port "+ str(port) + common_args
    print(train_cmd)
    if os.path.exists(os.path.join(output_path, "point_cloud/iteration_30000/point_cloud.ply")):
        print(f"The high resolution for {scene_path} has been created. Skip train")
        return output_path
    else:
        os.system(train_cmd)

    return output_path


def interpolate_poses(quats, tx, ty, tz, num_interpolations=5):
    # 初始化存储插值结果的列表
    interpolated_quats = []
    interpolated_tx = []
    interpolated_ty = []
    interpolated_tz = []

    # 遍历每一对原始位姿
    for i in range(len(quats) - 1):
        # 首先添加当前位姿
        interpolated_quats.append(quats[i])
        interpolated_tx.append(tx[i])
        interpolated_ty.append(ty[i])
        interpolated_tz.append(tz[i])

        # 准备进行插值
        key_rots = R.from_quat([quats[i], quats[i + 1]])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        times = np.linspace(0, 1, num=num_interpolations + 2)[1:-1]  # 排除原点，只取中间点

        # 执行插值
        interp_rots = slerp(times)
        interpolated_quats.extend(interp_rots.as_quat())

        # 对平移向量进行线性插值
        tx_interp = interp1d([0, 1], [tx[i], tx[i + 1]], kind='linear')(times)
        ty_interp = interp1d([0, 1], [ty[i], ty[i + 1]], kind='linear')(times)
        tz_interp = interp1d([0, 1], [tz[i], tz[i + 1]], kind='linear')(times)

        # 添加插值的平移向量
        interpolated_tx.extend(tx_interp)
        interpolated_ty.extend(ty_interp)
        interpolated_tz.extend(tz_interp)

    # 添加序列中的最后一个位姿
    interpolated_quats.append(quats[-1])
    interpolated_tx.append(tx[-1])
    interpolated_ty.append(ty[-1])
    interpolated_tz.append(tz[-1])

    return interpolated_quats, interpolated_tx, interpolated_ty, interpolated_tz

def cam_selection(train_cam_infos, train_cam_infos_sorted, rewrite=False):
    interpolated_images = []
    ref_names = {img.name for img in train_cam_infos}
    ref_indices = [i for i, img in enumerate(train_cam_infos_sorted) if img.name in ref_names]
    img_id = 1
    for start, end in zip(ref_indices, ref_indices[1:] + [ref_indices[-1] + 1]):
        segment = train_cam_infos_sorted[start:end]
        for i, img in enumerate(segment):
            if i == 0 and rewrite:
                base, ext = os.path.splitext(img.name)
                img_name = f'{base}_ref{ext}'
                new_image = BaseImage(
                    id=img_id,
                    qvec=img.qvec,
                    tvec=img.tvec,
                    camera_id=img.camera_id,
                    name=img_name,
                    xys=np.array([]),
                    point3D_ids=np.array([])
                )
            else:
                new_image = BaseImage(
                    id=img_id,
                    qvec=img.qvec,
                    tvec=img.tvec,
                    camera_id=img.camera_id,
                    name=img.name,
                    xys=np.array([]),
                    point3D_ids=np.array([])
                )
            interpolated_images.append(new_image)
            img_id += 1
    return {image.id: image for image in interpolated_images}



def fit_trajectory(train_cam_infos, use_lr, num_sample):
    num_inter=2
    
    # DO not need so dense sample points when num of sample is large
    if use_lr:
        if num_sample == 3:
            num_inter = 50
        elif num_sample <20:
            num_inter = 27
        else:
            num_inter = 10

    quats = [image.qvec for image in train_cam_infos]
    tx = [image.tvec[0] for image in train_cam_infos]
    ty = [image.tvec[1] for image in train_cam_infos]
    tz = [image.tvec[2] for image in train_cam_infos]
    interpolated_quats, interpolated_tx, interpolated_ty, interpolated_tz = interpolate_poses(quats, tx, ty, tz,
                                                                                              num_interpolations=num_inter)

    interpolated_images = [] # 新图像的起始ID
    camera_id = 1

    for i, (qvec, tvec_x, tvec_y, tvec_z) in enumerate(
            zip(interpolated_quats, interpolated_tx, interpolated_ty, interpolated_tz)):
        name = f"{i+1:04}.png" if i % (num_inter+1) != 0 else f"{i+1:04}_{train_cam_infos[i // (num_inter+1) ].name}.png"
        new_image = BaseImage(
            id=+i+1,
            qvec=qvec,
            tvec=np.array([tvec_x, tvec_y, tvec_z]),
            camera_id=camera_id,
            name=name,
            xys=np.array([]),
            point3D_ids=np.array([])
        )
        interpolated_images.append(new_image)


    return {image.id: image for image in interpolated_images}

def replace_last_directory(model_path, old, new):
    head, tail = os.path.split(model_path)
    modified_tail = tail.replace(old, new)
    head, tail = os.path.split(head)
    new_path = os.path.join(head, modified_tail)
    
    return new_path

def render(scene_path, model_path,  num_sample):
    resolution = 1
    cmd = f"python render.py -s {scene_path} -m {model_path} -r {resolution} --n_sparse {num_sample} --skip_test --load_custom --quiet"
    print(cmd)
    os.system(cmd)

    # cmd = f"python render.py -s {scene_path} -m {replace_last_directory(model_path, str(num_sample), 'hr')} -r {resolution} --n_sparse {num_sample} --skip_test --load_custom --quiet"
    # print(cmd)
    # os.system(cmd)

    return

def render_lr_hr_pairs(scene_path, lr_path, num_sample, use_lr):
    llffhold = 8
    cameras, images = read_model(path=os.path.join(scene_path, "colmap/sparse/0"), ext=".bin")
    train_cam_infos = [c for idx, c in images.items()]
    train_cam_infos_sorted = sorted(train_cam_infos.copy(), key=lambda x: x.name)

    ref_cameras, ref_images = read_model(path=os.path.join(lr_path, "sparse/0"), ext=".txt")
    ref_train_cam_infos = [c for idx, c in ref_images.items()]
    ref_train_cam_infos_sorted = sorted(ref_train_cam_infos.copy(), key=lambda x: x.name)

    test_cam_infos = [c for idx, c in images.items() if (idx-1) % llffhold == 0]
    test_cam_infos_sorted = sorted(test_cam_infos.copy(), key=lambda x: x.name)
    if num_sample > 0:
        if os.path.exists(os.path.join(scene_path, f"train_test_split_{str(num_sample)}.json")):
            json_path = os.path.join(scene_path, f"train_test_split_{str(num_sample)}.json")
            with open(json_path, "r") as f:
                idx_train = json.load(f)["train_ids"]
            train_cam_infos = [c for idx, c in images.items() if idx-1 in idx_train]
        else:
            idx = list(range(len(train_cam_infos)))
            idx_train = np.linspace(0, len(train_cam_infos) - 1, num_sample)
            idx_train = [round(i) for i in idx_train]
            train_cam_infos = [c for idx, c in enumerate(train_cam_infos_sorted) if idx in idx_train]

    if use_lr:
        # lr cams
        train_cam_infos = train_cam_infos
    else:
        # hr cams
        train_cam_infos = train_cam_infos_sorted

    selected_images = cam_selection(ref_train_cam_infos_sorted, train_cam_infos_sorted, True)
    # rewrite_selected_images = cam_selection(ref_train_cam_infos_sorted, train_cam_infos_sorted, True)
    # interpolated_images = fit_trajectory(train_cam_infos, use_lr, num_sample)

    render_cam_path = os.path.join(lr_path, "sparse/0_render")
    os.makedirs(render_cam_path, exist_ok=True)
    write_model(cameras, selected_images, render_cam_path, ext=".txt")

    # render_cam_path = os.path.join(lr_path, "sparse/0_final")
    # os.makedirs(render_cam_path, exist_ok=True)
    # write_model(cameras, rewrite_selected_images, render_cam_path, ext=".txt")
    # write_model(cameras, interpolated_images, render_cam_path, ext=".txt")

    render(scene_path, lr_path, num_sample)



def train_render_gss(scene_path, output_path, num_samples, use_lr, port):
    # if check_valid(scene_path):
    #     print(f"{scene_path} is valid")
    for num_sample in num_samples:
        lr_path = train_lr(scene_path, output_path, num_sample, port=port, eval_mode=False)
        render_lr_hr_pairs(scene_path,lr_path, num_sample, use_lr)
    # else:
    #     print(f"{scene_path} is not valid")




import random
import time
import os
import shutil
import argparse
import math
from PIL import Image
from torchvision.transforms import ToPILImage

import numpy as np
import warp as wp
import taichi as ti
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.mpm_core import MPMModel, set_boundary_conditions
from src.utils.render_utils import *
from src.utils.misc_utils import *
from src.utils.camera_view_utils import load_camera_params

def init_training(cfg, args=None):

    export_path = cfg.train.export_path if cfg.train.export_path else './outputs'
    if cfg.train.train_tag is None:
        cfg.train.train_tag = time.strftime("%Y%m%d_%H_%M_%S")
    export_path = os.path.join(export_path, cfg.train.train_tag)
    if not os.path.exists(export_path):
        os.makedirs(export_path)
        os.makedirs(os.path.join(export_path, 'images'))
        os.makedirs(os.path.join(export_path, 'videos'))
        os.makedirs(os.path.join(export_path, 'checkpoints'))

    seed = cfg.train.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = f'cuda:{cfg.train.gpu}'
    wp.init()
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})

    if cfg.preprocessing.particle_filling is not None:
        ti.init(arch=ti.cuda, device_memory_GB=8.0)

    torch_device = torch.device(device)
    torch.cuda.set_device(cfg.train.gpu)
    torch.backends.cudnn.benchmark = False
    print(f'\n device: {device}\n')

    print(f'exporting to: {export_path}\n')
    with open(os.path.join(export_path, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    writer_path = os.path.join(export_path, 'writers', 'writer_' + time.strftime("%Y%m%d_%H_%M_%S"))
    os.makedirs(writer_path)
    writer = SummaryWriter(writer_path)

    return torch_device, export_path, writer

def main(cfg, args=None):

    train_params = cfg.train
    preprocessing_params = cfg.preprocessing
    render_params = cfg.render
    material_params = cfg.material
    sim_params = cfg.sim

    text_descriptions = train_params.prompt

    torch_device, export_path, writer = init_training(cfg, args)

    to_pil = ToPILImage()

    print("Initializing gaussian and pre-processing...")
    model_path = train_params.model_path
    gaussians = load_gaussian_ckpt(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device=torch_device)
        if render_params.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device=torch_device)
    )
    (
        mpm_params,
        unselected_params,
        translate_params,
        screen_points,
        labels
    ) = load_params(
        gaussians, pipeline,
        preprocessing_params, export_path=export_path,
        obj_idxs=sim_params.obj_idx, fill_nums=sim_params.fill_nums
    )

    trans_pos = mpm_params['pos']
    trans_cov = mpm_params['cov']
    trans_opacity = mpm_params['opacity']
    trans_shs = mpm_params['shs']

    rotation_matrices = translate_params['rotation_matrices']
    scale_origin = translate_params['scale_origin']
    original_mean_pos = translate_params['original_mean_pos']

    gs_num = trans_pos.shape[0]

    print(f'Gaussian particle number: {gs_num}\n')

    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = load_camera_params(render_params, rotation_matrices, scale_origin, original_mean_pos)

    print(f'Exporting static gaussian rendering to {export_path}/static.png\n')
    export_static_gaussian_rendering(
        trans_pos, trans_cov, trans_shs, trans_opacity,
        unselected_params, rotation_matrices, scale_origin, original_mean_pos,
        model_path, pipeline, render_params, viewpoint_center_worldspace, observant_coordinates, gaussians, background, screen_points,
        export_path
    )

    print('Building MPM simulator and setting boundary conditions\n')
    mpm_model = MPMModel(
        sim_params,
        material_params,
        init_pos=trans_pos,
        enable_train=train_params.enable_train,
        device=torch_device
    )

    set_boundary_conditions(mpm_model, sim_params.boundary_conditions)

    print('Loading elasticity and plasticity model\n')

    elasticity, plasticity = init_constitute(
        material_params.elasticity,
        material_params.plasticity,
        elasticity_physicals=material_params.elasticity_physicals,
        plasticity_physicals=material_params.plasticity_physicals,
        requires_grad=train_params.enable_train,
        device=torch_device,
        obj_idx=sim_params.obj_idx
    )
    save_model_dict_arch(
        {
            'elasticity': elasticity,
            'plasticity': plasticity,
        },
        export_path
    )

    num_frames = sim_params.num_frames
    steps_per_frame = sim_params.steps_per_frame

    load_material_checkpoint(
        ckpt_dir=os.path.join(export_path, 'checkpoints'),
        epoch=train_params.ckpt_epoch,
        device=torch_device,
    )

    e_values = sim_params.e_values
    initial_e_value = torch.Tensor(e_values).log()
    log_E = nn.Parameter(initial_e_value.clone(), requires_grad=True)
    optimizer = torch.optim.Adam([log_E], lr=1e-3)
    print("Initial Youngs E, Log_e:", e_values, initial_e_value)

    requires_grad = False

    x_ckpt_ori = trans_pos.detach()
    v_ckpt_ori = torch.stack([torch.tensor([0.0, 0.0, -0.0001], device=torch_device) for _ in range(gs_num)])
    C_ckpt_ori = torch.zeros((gs_num, 3, 3), device=torch_device)
    F_ckpt_ori = torch.eye(3, device=torch_device).unsqueeze(0).repeat(gs_num, 1, 1)

    for epoch in range(train_params.epochs):
        x_ckpt = x_ckpt_ori.detach()
        v_ckpt = v_ckpt_ori.detach()
        C_ckpt = C_ckpt_ori.detach()
        F_ckpt = F_ckpt_ori.detach()

        mpm_model.reset()

        trans_pos = trans_pos.detach().requires_grad_(requires_grad)
        trans_cov = trans_cov.detach().requires_grad_(requires_grad)
        trans_shs = trans_shs.detach().requires_grad_(requires_grad)
        trans_opacity = trans_opacity.detach().requires_grad_(requires_grad)

        for i in unselected_params:
            if unselected_params[i] is not None:
                unselected_params[i] = unselected_params[i].detach()

        scale_origin = scale_origin.detach()
        original_mean_pos = original_mean_pos.detach()
        screen_points = screen_points.detach()

        important_frames = sorted({0, num_frames // 2, num_frames - 1})

        x = x_ckpt.clone().requires_grad_(False)
        v = v_ckpt.clone().requires_grad_(False)
        C = C_ckpt.clone().requires_grad_(False)
        F = F_ckpt.clone().requires_grad_(False)

        frame_data = []
        for frame in tqdm(range(num_frames), desc='Frames'):
            with torch.no_grad():
                frame_id = frame
                (
                    render_pos,
                    render_cov,
                    render_shs,
                    render_opacity,
                    render_rot
                ) = get_mpm_gaussian_params(
                    pos=x, cov=trans_cov, shs=trans_shs, opacity=trans_opacity,
                    F=F,
                    unselected_params=unselected_params,
                    rotation_matrices=rotation_matrices,
                    scale_origin=scale_origin,
                    original_mean_pos=original_mean_pos
                )

                rendering = render_mpm_gaussian(
                    model_path=model_path,
                    pipeline=pipeline,
                    render_params=render_params,
                    step=frame_id,
                    viewpoint_center_worldspace=viewpoint_center_worldspace,
                    observant_coordinates=observant_coordinates,
                    gaussians=gaussians,
                    background=background,
                    pos=render_pos,
                    cov=render_cov,
                    shs=render_shs,
                    opacity=render_opacity,
                    rot=render_rot,
                    screen_points=screen_points,
                )
                if train_params.export_video:
                    export_rendering(rendering, frame_id, folder=os.path.join(export_path, 'images'))

                if frame in important_frames:

                    frame_data.append({
                        'x': x.clone(),
                        'v': v.clone(),
                        'C': C.clone(),
                        'F': F.clone()
                    })

            for step in range(steps_per_frame):
                with torch.no_grad():
                    stress = checkpoint(elasticity, F, labels, log_E.detach())
                    assert torch.all(torch.isfinite(stress))

                    x, v, C, F = checkpoint(mpm_model, x, v, C, F, stress)
                    assert torch.all(torch.isfinite(x))

                    F = checkpoint(plasticity, F, labels, log_E.detach())
                    assert torch.all(torch.isfinite(F))

            torch.cuda.empty_cache()

        defo_target = 0.1
        E_REF = 3e6
        with torch.no_grad():
            eta = (0.1 * torch.log1p(log_E.detach().exp()) / math.log1p(E_REF)).clamp(max=1.0)

        if frame_data:
            optimizer.zero_grad()

            total_grad = torch.zeros_like(log_E)
            grad_count = 0

            for data in frame_data:
                x_val = data['x'].requires_grad_(False)
                v_val = data['v'].requires_grad_(False)
                C_val = data['C'].requires_grad_(False)
                F_val = data['F'].requires_grad_(True)

                with torch.enable_grad():
                    stress_keyframe = elasticity(F_val, labels, log_E)
                    x_, v_, C_, F_ = mpm_model(x_val, v_val, C_val, F_val, stress_keyframe)
                    F_ = plasticity(F_, labels, log_E)

                    stress_norm = torch.norm(stress_keyframe)
                    grad_stress = torch.autograd.grad(stress_norm, log_E, retain_graph=True)[0]
                    pseudo_grad = torch.log1p(grad_stress.abs())

                    I = torch.eye(3, device=F_.device, dtype=F_.dtype)
                    F_dev = F_ - I
                    frob_norm = torch.norm(F_dev, p='fro', dim=(-2, -1)).mean()
                    defo_error = frob_norm - defo_target

                    pseudo_defo = (18.0 * defo_error).to(log_E.device)

                    total_grad += eta * pseudo_grad - (1 - eta) * pseudo_defo
                    grad_count += 1

                    optimizer.zero_grad()

            avg_grad = total_grad / grad_count
            print(' log_E old', log_E)
            print(' eta:', eta)
            print(' grad_stress:', grad_stress)
            print(' pseudo grad:', pseudo_grad)
            print(' defo_error:', defo_error)
            print(' pesudo_defo:', pseudo_defo)

            with torch.no_grad():
                log_E.data = log_E.data - avg_grad

            print(f" log_E update: {log_E}")

        save_video(f'{export_path}/images',f'{export_path}/videos/video_{epoch:04d}.mp4')
        print(f'Training or rendering finished. The result has been exported to {export_path}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file."
    )

    parser.add_argument(
        "--test",
        action='store_true',
        help="Test mode."
    )

    parser.add_argument(
        "--gpu",
        type=int,
        help="GPU index."
    )

    parser.add_argument(
        "--tag",
        type=str,
        help="Training tag."
    )

    parser.add_argument(
        "--overwrite",
        '-o',
        action='store_true',
        help="Overwrite the existing export folder."
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output folder."
    )

    parser.add_argument(
        "--save_internal",
        action='store_true',
        help="Save internal checkpoints."
    )

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    if args.gpu is not None:
        cfg.train.gpu = args.gpu
    if args.test:
        cfg.train.enable_train = False
    if args.tag is not None:
        cfg.train.train_tag = args.tag
    if args.output is not None:
        cfg.train.export_path = args.output

    main(cfg, args)

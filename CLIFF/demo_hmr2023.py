"""
ProHMR demo script.
To run our method you need a folder with images and corresponding OpenPose detections.
These are used to crop the images around the humans and optionally to fit the SMPL model on the detections.

Example usage:
python demo.py --checkpoint=path/to/checkpoint.pt --img_folder=/path/to/images --keypoint_folder=/path/to/json --out_folder=/path/to/output --run_fitting

Running the above will run inference for all images in /path/to/images with corresponding keypoint detections.
The rendered results will be saved to /path/to/output, with the suffix _regression.jpg for the regression (mode) and _fitting.jpg for the fitting.

Please keep in mind that we do not recommend to use `--full_frame` when the image resolution is above 2K because of known issues with the data term of SMPLify.
In these cases you can resize all images such that the maximum image dimension is at most 2K.
"""
from pathlib import Path
import torch
import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np

from hmr2023.configs import get_config, prohmr_config, dataset_config, dataset_config_old, prohmr_config
from hmr2023.datasets import create_dataset, create_webdataset
from hmr2023.models import ProHMR
from hmr2023.optimization import KeypointFitting
from hmr2023.utils import recursive_to
from hmr2023.datasets import OpenPoseDataset
from hmr2023.utils.renderer import Renderer

parser = argparse.ArgumentParser(description='ProHMR demo code')
parser.add_argument('--checkpoint', type=str, default='data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt', help='Path to pretrained model checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
parser.add_argument('--img_folder', type=str, default=None, help='Folder with input images')
parser.add_argument('--dataset_name', type=str, default=None, help='Name of the dataset. If not set, use the specified keypoint_folder')
parser.add_argument('--keypoint_folder', type=str, default=None, help='Folder with corresponding OpenPose detections')
parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
parser.add_argument('--out_format', type=str, default='jpg', choices=['jpg', 'png'], help='Output image format')
parser.add_argument('--run_fitting', dest='run_fitting', action='store_true', default=False, help='If set, run fitting on top of regression')
parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, run fitting in the original image space and not in the crop.')
parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
parser.add_argument('--naive_focal_cliff', dest='naive_focal_cliff', action='store_true', default=False)


args = parser.parse_args()

# Use the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if False: # PROHMR
    if args.model_cfg is None:
        args.model_cfg = str(Path(args.checkpoint).parent.parent / 'model_config.yaml')
    model_cfg = get_config(args.model_cfg)

    # Setup model
    model = ProHMR.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg).to(device)
    model.eval()
else:
    # Dummy model config. Hardcode image size for CLIFF
    model_cfg = prohmr_config()
    model_cfg.defrost()
    model_cfg.MODEL.IMAGE_SIZE=224
    model_cfg.freeze()

    # from core import path_config
    # from models.CLIFF_net_wrapper import CLIFF_Tester
    # model = CLIFF_Tester(path_config.SMPL_MEAN_PARAMS, args.checkpoint)
    from cliff_wrapper import CLIFFWrapper2023
    model = CLIFFWrapper2023(args.checkpoint)
    model = model.to(device)

def get_dataset_config(name):
    try:
        dataset_cfg = dataset_config()[name]

        # Read from /fsx directly for eval, don't copy all data to /scratch
        if dataset_cfg.URLS.startswith('/scratch/shubham/data/hmr2023_data_shuffled'):
            # Replace scratch with fsx
            dataset_cfg.defrost()
            dataset_cfg.URLS = dataset_cfg.URLS.replace(
                '/scratch/shubham/data/hmr2023_data_shuffled',
                '/fsx/shubham/data/hmr2023_data_shuffled'
            )
            dataset_cfg.freeze()
    except KeyError:
        dataset_cfg = dataset_config_old()[name]
    return dataset_cfg

# Create a dataset on-the-fly
if args.dataset_name is not None:
    dataset_cfg = get_dataset_config(args.dataset_name)
    try:
        dataset = create_webdataset(model_cfg, dataset_cfg, train=False)
    except:
        dataset = create_dataset(model_cfg, dataset_cfg, train=False)
else:
    if None in [args.img_folder, args.keypoint_folder]:
        raise ValueError('If dataset_name is not specified, img_folder and keypoint_folder must be specified.')
    dataset = OpenPoseDataset(model_cfg, img_folder=args.img_folder, keypoint_folder=args.keypoint_folder, max_people_per_image=1)

# Setup a dataloader with batch_size = 1 (Process images sequentially)
dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False)

# Setup the renderer
renderer = Renderer(model_cfg, faces=model.smpl.faces)

# Make directory if it does not exist
os.makedirs(args.out_folder, exist_ok=True)

# Go over each image in the dataset
for i, batch in enumerate(tqdm(dataloader)):

    batch['box_center'].shape
    batch['box_size'].shape
    batch['img_size'].shape

    batch = recursive_to(batch, device)
    with torch.no_grad():
        out, _ = model(batch, naive_focal_cliff=args.naive_focal_cliff)

    batch_size = batch['img'].shape[0]
    for n in range(batch_size):
        img_fn, _ = os.path.splitext(batch['imgname_rel'][n])
        person_id = int(batch['personid'][n])
        regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                  out['pred_cam_t'][n].detach().cpu().numpy(),
                                  batch['img'][n])
        if args.side_view:
            side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                    out['pred_cam_t'][n].detach().cpu().numpy(),
                                    batch['img'][n],
                                    side_view=True)
            regression_img = np.concatenate([regression_img, side_img], axis=1)

        cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.{args.out_format}'), 255*regression_img[:, :, ::-1])

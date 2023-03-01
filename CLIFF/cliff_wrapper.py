import torch
from loguru import logger


from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from models.cliff_res50.cliff import CLIFF as cliff_res50
from common import constants
from common.utils import strip_prefix_if_present, cam_crop2full, video_to_images
from common.utils import estimate_focal_length
from common.renderer_pyrd import Renderer
from lib.yolov3_detector import HumanDetector
from common.mocap_dataset import MocapDataset
from lib.yolov3_dataset import DetectionDataset


MIN_NUM_FRAMES = 0


class CLIFFWrapper2023:
    def __init__(self, ckpt):

        # Create the model instance
        # cliff = eval("cliff_" + args.backbone)
        cliff = cliff_hr48
        self.model = cliff(constants.SMPL_MEAN_PARAMS)
        # Load the pretrained model
        print("Load the CLIFF checkpoint from path:", ckpt)
        state_dict = torch.load(ckpt, map_location='cpu')['model']
        state_dict = strip_prefix_if_present(state_dict, prefix="module.")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        
        # Other stuff
        from hmr2023.models.smpl_wrapper import SMPL as SMPL_hmr
        HMR_ROOT = '/fsx/shubham/code/hmr2023'
        # SMPL_DIR = f'{HMR_ROOT}/data/smpl'
        SMPL_DIR = '/fsx/shubham/code/PARE/data/body_models/smpl/'
        self.smpl = SMPL_hmr(SMPL_DIR, 
                            gender='neutral', 
                            joint_regressor_extra=f'{HMR_ROOT}/data/SMPL_to_J19.pkl')


    def to(self, device):
        self.model = self.model.to(device)
        self.smpl = self.smpl.to(device)
        return self

    @torch.no_grad()
    def __call__(self, batch, naive_focal_cliff=False):
        focal_length = 5000.
        image_size = 224.

        # Forward pass on CLIFF model
        norm_img = batch["img"].float()
        assert norm_img.shape[1:] == (3, image_size, image_size)

        # What's available in our batch?
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        
        # Convert to CLIFF format:
        #   center = batch["center"].float()
        #   scale = batch["scale"].float()
        #   img_h = batch["img_h"].float()
        #   img_w = batch["img_w"].float()
        img_w = img_size[:, 0]
        img_h = img_size[:, 1]
        center = box_center
        scale = box_size / 200.0

        # focal_length = batch["focal_length"].float()

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        # The constants below are used for normalization, and calculated from H36M data.
        # It should be fine if you use the plain Equation (5) in the paper.
        focal_length_cliff = (img_h**2 + img_w**2).sqrt()
        if naive_focal_cliff:
            bbox_info[:, :2] = bbox_info[:, :2] / focal_length_cliff.unsqueeze(-1)  # [-1, 1]
            bbox_info[:, 2] = bbox_info[:, 2] / focal_length_cliff  # [-1, 1]
        else:
            bbox_info[:, :2] = bbox_info[:, :2] / focal_length_cliff.unsqueeze(-1) * 2.8  # [-1, 1]
            bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length_cliff) / (0.06 * focal_length_cliff)  # [-1, 1]

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_cam_crop = self.model(norm_img, bbox_info)

        new_output = {
            'pred_cam': pred_cam_crop,
            'pred_smpl_params': {
                'global_orient': pred_rotmat[:,0:1,:,:], # B,1,3,3
                'body_pose': pred_rotmat[:,1:,:,:], # B,23,3,3
                'betas': pred_betas,
            },
        }

        # Convert outputs to HMR2023 format
        pred_cam = new_output['pred_cam']
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                pred_cam[:, 2],
                                2 * focal_length / (image_size * pred_cam[:, 0] + 1e-9)], dim=-1)

        # Recompute 3D keypoints using HMR2023 joint regressor
        pred_smpl_out = self.smpl(
            global_orient=new_output['pred_smpl_params']['global_orient'],
            body_pose=new_output['pred_smpl_params']['body_pose'],
            betas=new_output['pred_smpl_params']['betas'],
        )
        pred_keypoints_3d = pred_smpl_out.joints
        pred_vertices = pred_smpl_out.vertices

        # Recompute 2D keypoints
        from hmr2023.utils.geometry import perspective_projection
        pred_keypoints_2d = perspective_projection(
                pred_keypoints_3d,
                translation = pred_cam_t,
                focal_length = focal_length * torch.ones_like(pred_cam[:,:2]),
            )
        pred_keypoints_2d = pred_keypoints_2d / image_size

        new_output.update(
            pred_cam_t = pred_cam_t,
            pred_vertices = pred_vertices,
            pred_keypoints_3d = pred_keypoints_3d[:,:44,:],
            pred_keypoints_2d = pred_keypoints_2d[:,:44,:],
            focal_length = focal_length * torch.ones_like(pred_cam[:,:2]),
        )

        return new_output, None

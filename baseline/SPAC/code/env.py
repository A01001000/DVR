import torch
import torchvision.utils as vutils
import SimpleITK as sitk
import numpy as np
import cv2
import copy
from scipy import misc
from random import shuffle
from config import Config as cfg
from networks import SpatialTransformer
import utils
import os
from utils import MINDLoss

idx = 64

class Env(object):
    """
    The environment must specify a root directory including paired files in CT and MR respectively
    The virtual_label is used to assist the registration learning process
    """
    def __init__(self, generator, stn, seg_stn, use_seg=False, device=None):
        # self.generator = self._generator(dataset)
        self.generator = generator
        self.stn = stn
        self.seg_stn = seg_stn
        self.use_seg = use_seg
        self.device = device
        # self.labels = utils.brain_labels()
        self.reset()


    def _generator(self, datasets):
        indices = np.arange(len(datasets))
        self.epoch = 0
        while True:
            np.random.shuffle(indices)
            self.epoch += 1
            for idx in indices:
                items = datasets.getitem(idx)
                yield items

    def reset(self):

        items = next(self.generator)
        # print(items)

        self.tensor_fixed = utils.tensor(items['fixed']).to(self.device)
        self.tensor_moving = utils.tensor(items['moving']).to(self.device)
        self.tensor_moved = copy.deepcopy(self.tensor_moving)

        if self.use_seg:
            self.fixed_seg = items['fixed_seg']
            # self.labels = np.unique(self.fixed_seg)
            self.moving_seg = utils.tensor(items['moving_seg']).to(self.device)

            # self.moving_seg = []
            # for i, lab in enumerate(self.labels):
            #     m_seg = (m_segs == lab)*255.
            #     self.moving_seg.append(m_seg)

            self.moved_seg = copy.deepcopy(self.moving_seg)
        else:
            # Initialize segmentation attributes as None when not using segmentation
            self.fixed_seg = None
            self.moving_seg = None
            self.moved_seg = None

        self.prev_score = self.score()
        self.field = None
        self.global_step = 1
        return self.state(), self.prev_score

    def state(self):
        return torch.cat([self.tensor_fixed, self.tensor_moved], dim=1).to(self.device)

    def score(self):
        if self.use_seg:
            # dices = []
            # for i, lab in enumerate(self.labels):
            #     f_seg = (self.fixed_seg == lab).astype(np.uint8)
            #     m_seg = utils.numpy_im(self.moved_seg[i] > 0, 1, device=self.device)

            #     dice = utils.dice_(f_seg, m_seg, [1])
            #     dices.append(dice)
            dices = utils.dice(self.fixed_seg[0, 0]>0,
                utils.numpy_im(self.moved_seg, 1, device=self.device)>0)
            return np.mean(dices)
        else:
            # --- Use MIND loss as reward ---
            try:
                mind_loss_fn = MINDLoss()
                mind_loss = mind_loss_fn(self.tensor_moved, self.tensor_fixed)
                mind_score = -mind_loss.cpu().item()
                print(f"[DEBUG] MIND loss: {mind_loss.cpu().item()}")
                return mind_score
            except Exception as e:
                print(f"Error calculating MIND score: {e}")
                return 0.0


    def done(self):
        return self.score() > cfg.SCORE_THRESHOLD

    # latent is a tensor representing the displacement field
    def step(self, field, global_ep):
        # We will only save debug images for one specific episode (e.g., 101)
        if global_ep == 101:
            debug_dir = os.path.join(cfg.PROCESS_PATH, f"ep_{global_ep}_debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # 1. Visualize the input deformation field for this step
            field_slice = field[0, 0, :, :, idx].detach().cpu().numpy()
            field_vis = cv2.normalize(field_slice, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imwrite(os.path.join(debug_dir, f"step_{self.global_step}_0_input_field.png"), field_vis)

        # Correctly accumulate the deformation field
        if self.field is None:
            self.field = field
        else:
            self.field = self.stn(self.field, field) + field
        
        # Apply the total accumulated deformation
        self.tensor_moved = self.stn(self.tensor_moving, self.field)
        
        if global_ep == 101:
            # 2. Visualize the images: Fixed | Original Moving | Warped Moving
            fixed_vis = self.tensor_fixed[0, 0, :, :, idx].detach().cpu().numpy() * 255
            moving_vis = self.tensor_moving[0, 0, :, :, idx].detach().cpu().numpy() * 255
            moved_vis = self.tensor_moved[0, 0, :, :, idx].detach().cpu().numpy() * 255
            comparison_img = np.hstack([fixed_vis, moving_vis, moved_vis]).astype(np.uint8)
            cv2.imwrite(os.path.join(debug_dir, f"step_{self.global_step}_1_images.png"), comparison_img)

        self.global_step += 1
        
        current_score = self.score()
        
        # Use a dense reward system
        reward = current_score
        
        done = current_score > cfg.SCORE_THRESHOLD
        if done:
            reward += 10 # Add a bonus for success

        # --- DEBUG: Print segmentation stats before reward ---
        if hasattr(self, 'moving_seg') and hasattr(self, 'fixed_seg'):
            print(f"[DEBUG] moving_seg sum: {np.sum(self.moving_seg)}, unique: {np.unique(self.moving_seg)}")
            print(f"[DEBUG] fixed_seg sum: {np.sum(self.fixed_seg)}, unique: {np.unique(self.fixed_seg)}")

        # --- DEBUG: Print Dice or reward metric if available ---
        if self.use_seg:
            print(f"[DEBUG] Dice score: {current_score}")
        else:
            print(f"[DEBUG] MIND-based reward: {current_score}")
        print(f"[DEBUG] Reward: {reward}")

        return reward, self.state(), done, current_score

    def save_init(self, dir=cfg.PROCESS_PATH):
        vutils.save_image(self.tensor_fixed[:, :, :, idx, :].data, dir+'/fixed.bmp', normalize=True)
        vutils.save_image(self.tensor_moving[:, :, :, idx, :].data, dir+'/moving.bmp', normalize=True)

    def save_process(self, index, dir=cfg.PROCESS_PATH):
        if index % 3 == 0:
            # tmp_field = utils.render_flow(utils.numpy_im(self.field, 1., self.device)[:, :, idx, :])
            # cv2.imwrite('{}/field-{}.png'.format(dir, index), tmp_field)
            vutils.save_image(self.tensor_moved[:, :, :, idx, :].data, dir+'/moved-{}.bmp'.format(index), normalize=True)



























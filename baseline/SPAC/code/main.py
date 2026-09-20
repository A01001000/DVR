print("Script started")
import torch
print("Imported torch")
import random
import numpy as np
import copy
import argparse
import os

import utils
import data_util
import data_util.brain
import data_util.liver
import data_util.custom
from data_util.data import Split
from dataloader import BrainData
from config import Config as cfg
from brain import SPAC
from env import Env
from agent import Agent
from summary import Summary
from networks import *
from config import config

import sys
sys.stdout.flush()

# os.environ['CUDA_VISIBLE_DEVICES'] = pa.GPU_ID

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(cfg.GPU_ID)
else:
    device = torch.device('cpu')

# device = torch.device('cpu')
print("Imports done")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True # cpu\gpu 结果一致
    
def parse_args():
    parser = argparse.ArgumentParser(description='Train SPAC Model')
    parser.add_argument('--data_type', type=str, default=cfg.DATA_TYPE, help='Dataset type (mrxfdg, l2r, chaos)')
    parser.add_argument('--train_data', type=str, default=cfg.TRAIN_DATA, help='Path to training data')
    args = parser.parse_args()
    return args

def update_config(updates):
    """
    Updates attributes of the imported 'cfg' object directly in memory.

    Args:
        updates (dict): A dictionary with keys matching the attributes
                        to update and their new values.
    """
    for key, value in updates.items():
        # Check if the attribute exists before trying to set it
        if hasattr(cfg, key):
            setattr(cfg, key, value)
            print(f"Updated '{key}' to: {value}")
        else:
            print(f"Warning: Attribute '{key}' not found in config. Ignoring.")

if __name__ == "__main__":
    setup_seed(cfg.SEED)
    utils.mkdir(cfg.LOG_DIR)
    utils.remkdir(cfg.PROCESS_PATH)
    utils.mkdir(cfg.MODEL_PATH)

    summary = Summary(cfg.LOG_DIR)
    
    args = parse_args()
    updates = {
        "DATA_TYPE": args.data_type, # liver, brain, mrxfdg
        "TRAIN_DATA": args.train_data
    }
    update_config(updates)

    # --- Ensure segmentations are NOT used for training (only for evaluation/testing) ---
    cfg.USE_SEG = False  # Always use images only for training; set True for evaluation

    #######################################
    stn = SpatialTransformer(cfg.HEIGHT, 'bilinear').to(device) # nearest
    seg_stn = SpatialTransformer(cfg.HEIGHT, mode='nearest').to(device) # nearest

    # datasets = BrainData(cfg.TRAIN_DATA, cfg.TRAIN_SEG_DATA, cfg.ATLAS, use_seg=cfg.USE_SEG, mode='train', aug=cfg.BSPLINE_AUG)

    Dataset = eval('data_util.{}.Dataset'.format(cfg.IMAGE_TYPE if cfg.IMAGE_TYPE else 'custom'))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'datasets', f'{cfg.DATA_TYPE}.json')
    dataset = Dataset(split_path=dataset_path, paired=True, affine=False)

    # Explicitly request the 'train' subset from scheme '1'
    print("Creating training data generator...")
    generator = dataset.generator(subset_key='1', subset_name='train', batch_size=1, loop=True)


    brain = SPAC(stn, device)
    if cfg.PRE_TRAINED:
        print('pretrain models')
        brain.load_model('actor', cfg.ACTOR_MODEL)
        brain.load_model('decoder', cfg.DECODER_MODEL_RL)
        brain.load_model('critic1', cfg.CRITIC1_MODEL)
        brain.load_model('critic2', cfg.CRITIC2_MODEL)

    env = Env(generator, stn, seg_stn, cfg.USE_SEG, device)
    agent = Agent(brain, env, summary, device=device)

    agent.run()







#!/usr/bin/env python3
"""
Optimized training script for VoxelMorph and NestedMorph models with fixes for:
1. Regularization calculation bug
2. CSV logging
3. Better learning rate and weight settings
4. Early stopping
"""

import os
import sys
import glob
import time
import torch
import argparse
import numpy as np
import csv
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from torchvision import transforms
from natsort import natsorted
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nestedmorph.src.utils.utils import *
from nestedmorph.src.losses.losses import *
from nestedmorph.src.data.nifti_datasets import create_nifti_dataloaders
from nestedmorph.src.data.trans import *
from nestedmorph.src.utils.config import device

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Logger(object):
    """Custom logger class to redirect stdout to both terminal and log file."""
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters()) / 1e6


class EarlyStopping:
    """Early stopping utility."""
    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint."""
        self.best_weights = model.state_dict().copy()


def train_model(args):
    """
    Train the model for multimodal MR-CT registration with optimizations.
    """
    # Create experiment directory
    save_dir = f'{args.model}_optimized_{args.data_name}/'
    
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    
    # Redirect stdout to logger
    sys.stdout = Logger('logs/' + save_dir)
    
    print(f"Starting OPTIMIZED training for {args.model} with data: {args.data_dirs}")
    print(f"Arguments: {vars(args)}")
    
    # Initialize model
    if args.model == "NestedMorph":
        from nestedmorph.src.models.nestedmorph import NestedMorph
        model = NestedMorph(inshape=tuple(args.img_size))
    elif args.model == "VoxelMorph":
        from nestedmorph.src.models.voxelmorph import VoxelMorph
        model = VoxelMorph(
            inshape=tuple(args.img_size),
            nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model.to(device)
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params:.2f}M")
    
    # Initialize spatial transformation
    reg_model = register_model(args.img_size, 'nearest')
    reg_model.to(device)
    
    # Data transforms
    train_composed = transforms.Compose([
        NumpyType((np.float32, np.float32)),
    ])
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_nifti_dataloaders(
        train_dirs=args.data_dirs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        train_split=args.train_split,
        transforms=train_composed
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize optimizer with optimized settings
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, amsgrad=True)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Choose loss functions with better weights
    if args.data_name.lower() in ['mrct', 'mr_ct', 'multimodal', 'l2r_multiset']:
        print("Using Normalized Cross Correlation loss for multimodal MR-CT registration")
        criterion = NCCLoss()
        criterions = [criterion, FlowGrad3d(penalty='l2')]
        weights = [1, 0.1]  # Better regularization weight
    else:
        print("Using MIND loss for monomodal registration")
        criterion = MINDLoss()
        criterions = [criterion, FlowGrad3d(penalty='l2')]
        weights = [1, 0.1]
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    
    # Training setup
    best_dsc = 0
    train_losses = []
    val_dice_scores = []
    
    # CSV logging setup
    csv_file = f'experiments/{save_dir}/training_stats.csv'
    csv_columns = ['epoch', 'train_loss', 'val_loss', 'val_dice', 'learning_rate', 'sim_loss', 'reg_loss']
    
    with open(csv_file, 'w', newline='') as file:
        writer_csv = csv.DictWriter(file, fieldnames=csv_columns)
        writer_csv.writeheader()
    
    # TensorBoard logging
    writer = SummaryWriter(f'logs/{save_dir}/tensorboard')
    
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Early stopping patience: {args.patience} epochs")
    
    for epoch in range(args.epochs):
        '''Training phase'''
        model.train()
        epoch_loss = 0
        epoch_sim_loss = 0
        epoch_reg_loss = 0
        loss_all = AverageMeter()
        
        for batch_idx, (data) in enumerate(train_loader):
            # Move data to device
            x, y = data[0].to(device), data[1].to(device)  # moving, fixed
            
            # Forward pass
            optimizer.zero_grad()
            
            # Model inference
            output = model((x, y))
            
            # Handle different model outputs
            if isinstance(output, tuple):
                warped_image, flow = output
            else:
                flow = output
                warped_image = reg_model(x, flow)
            
            # Compute losses with FIXED calculation
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                if n == 0:  # Image similarity loss
                    curr_loss = loss_function(warped_image, y) * weights[n]
                else:  # Regularization loss (FIXED: only pass flow, not y)
                    curr_loss = loss_function(flow) * weights[n]
                
                loss_vals.append(curr_loss)
                loss += curr_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            loss_all.update(loss.item(), y.numel())
            epoch_loss += loss.item()
            epoch_sim_loss += loss_vals[0].item()
            epoch_reg_loss += loss_vals[1].item()
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(train_loader)}, '
                           f'Loss: {loss.item():.4f}, Sim: {loss_vals[0].item():.6f}, Reg: {loss_vals[1].item():.6f}')
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_sim_loss = epoch_sim_loss / len(train_loader)
        avg_reg_loss = epoch_reg_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        '''Validation phase'''
        model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for batch_idx, (data) in enumerate(val_loader):
                x, y = data[0].to(device), data[1].to(device)
                
                # Forward pass
                output = model((x, y))
                
                if isinstance(output, tuple):
                    warped_image, flow = output
                else:
                    flow = output
                    warped_image = reg_model(x, flow)
                
                # Compute validation loss with FIXED calculation
                loss = 0
                for n, loss_function in enumerate(criterions):
                    if n == 0:  # Image similarity loss
                        curr_loss = loss_function(warped_image, y) * weights[n]
                    else:  # Regularization loss (FIXED)
                        curr_loss = loss_function(flow) * weights[n]
                    loss += curr_loss
                
                val_loss += loss.item()
                
                # Compute Dice score for validation
                dice_score = dice(warped_image, y)
                val_dice += dice_score
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        val_dice_scores.append(avg_val_dice)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Loss/Similarity', avg_sim_loss, epoch)
        writer.add_scalar('Loss/Regularization', avg_reg_loss, epoch)
        writer.add_scalar('Dice/Validation', avg_val_dice, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log to CSV
        with open(csv_file, 'a', newline='') as file:
            writer_csv = csv.DictWriter(file, fieldnames=csv_columns)
            writer_csv.writerow({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_dice': avg_val_dice,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'sim_loss': avg_sim_loss,
                'reg_loss': avg_reg_loss
            })
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{args.epochs}: Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}, '
              f'Sim: {avg_sim_loss:.4f}, Reg: {avg_reg_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model
        if avg_val_dice > best_dsc:
            best_dsc = avg_val_dice
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),
            }, f'experiments/{save_dir}/best_model_{args.model}.pth.tar')
            print(f'New best model saved with Dice: {best_dsc:.4f}')
        
        # Early stopping check
        if early_stopping(avg_val_loss, model):
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
        
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),
            }, f'experiments/{save_dir}/checkpoint_epoch_{epoch+1}.pth.tar')
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'best_dsc': best_dsc,
        'optimizer': optimizer.state_dict(),
    }, f'experiments/{save_dir}/final_model_{args.model}.pth.tar')
    
    writer.close()
    print(f"Training completed! Best validation Dice: {best_dsc:.4f}")
    print(f"Models saved in: experiments/{save_dir}/")
    print(f"Training stats saved in: {csv_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train registration models on NIfTI data (OPTIMIZED)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='VoxelMorph',
                        choices=['VoxelMorph', 'NestedMorph'],
                        help='Model to train')
    
    # Data arguments
    parser.add_argument('--data-dirs', type=str, nargs='+', required=True,
                        help='List of training dataset directories')
    parser.add_argument('--data-name', type=str, default='L2R_multiset',
                        help='Dataset name for saving')
    parser.add_argument('--img-size', type=int, nargs=3, default=[128, 128, 128],
                        help='Image size (H W D)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Training split ratio')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=120,
                        help='Number of training epochs (reduced from 400)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_model(args)

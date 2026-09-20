#!/usr/bin/env python3
"""
Updated training script for VoxelMorph and NestedMorph models using NIfTI data format.

Usage:
    python train_nifti.py --model VoxelMorph --data-dirs ../datasets/Dataset1/ ../datasets/Dataset2/ --epochs 1000
    python train_nifti.py --model NestedMorph --data-dirs ../datasets/Dataset1/ --epochs 500 --lr 1e-4
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

from src.utils.utils import *
from src.losses.losses import *
from src.data.nifti_datasets import create_nifti_dataloaders
from src.data.trans import *
from src.utils.config import device

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


def train_model(args):
    """
    Train the model for multimodal MR-CT registration.
    """
    # Create experiment directory
    save_dir = f'{args.model}_nifti_{args.data_name}/'
    
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    
    # Redirect stdout to logger
    sys.stdout = Logger('logs/' + save_dir)
    
    print(f"Starting training for {args.model} with data: {args.data_dirs}")
    print(f"Arguments: {vars(args)}")
    
    # Initialize model
    if args.model == "NestedMorph":
        from src.models.nestedmorph import NestedMorph
        model = NestedMorph(inshape=tuple(args.img_size))
    elif args.model == "VoxelMorph":
        from src.models.voxelmorph import VoxelMorph
        model = VoxelMorph(
            inshape=tuple(args.img_size),  # Use actual image size from arguments
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
    
    # Data transforms (adapted for NIfTI 4D data)
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
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
    
    # Choose loss functions
    if args.data_name.lower() in ['mrct', 'mr_ct', 'multimodal']:
        print("Using Normalized Cross Correlation loss for multimodal MR-CT registration")
        criterion = NCCLoss()
        criterions = [criterion, FlowGrad3d(penalty='l2')]
        weights = [1, 0.1]  # Increased regularization weight
    else:
        print("Using MIND loss for monomodal registration")
        criterion = MINDLoss()
        criterions = [criterion, FlowGrad3d(penalty='l2')]
        weights = [1, 0.1]  # Increased regularization weight
    
    # Training setup
    best_dsc = 0
    train_losses = []
    val_dice_scores = []
    
    # CSV logging setup
    csv_file = f'experiments/{save_dir}/training_stats.csv'
    csv_columns = ['epoch', 'train_loss', 'val_loss', 'val_dice', 'learning_rate']
    
    with open(csv_file, 'w', newline='') as file:
        writer_csv = csv.DictWriter(file, fieldnames=csv_columns)
        writer_csv.writeheader()
    
    # TensorBoard logging
    writer = SummaryWriter(f'logs/{save_dir}/tensorboard')
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        '''Training phase'''
        model.train()
        epoch_loss = 0
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
            
            # Compute losses
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                if n == 0:  # Image similarity loss
                    curr_loss = loss_function(warped_image, y) * weights[n]
                else:  # Regularization loss (gradient smoothness)
                    curr_loss = loss_function(flow) * weights[n]
                
                loss_vals.append(curr_loss)
                loss += curr_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            loss_all.update(loss.item(), y.numel())
            epoch_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(train_loader)}, '
                           f'Loss: {loss.item():.4f}, Sim: {loss_vals[0].item():.6f}, Reg: {loss_vals[1].item():.6f}')
        
        avg_train_loss = epoch_loss / len(train_loader)
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
                
                # Compute validation loss
                loss = 0
                for n, loss_function in enumerate(criterions):
                    if n == 0:  # Image similarity loss
                        curr_loss = loss_function(warped_image, y) * weights[n]
                    else:  # Regularization loss
                        curr_loss = loss_function(flow) * weights[n]
                    loss += curr_loss
                
                val_loss += loss.item()
                
                # Compute Dice score for validation (simplified)
                dice_score = dice(warped_image, y)
                val_dice += dice_score
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        val_dice_scores.append(avg_val_dice)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Dice/Validation', avg_val_dice, epoch)
        
        # Log to CSV
        with open(csv_file, 'a', newline='') as file:
            writer_csv = csv.DictWriter(file, fieldnames=csv_columns)
            writer_csv.writerow({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_dice': avg_val_dice,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{args.epochs}: Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}')
        
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
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train registration models on NIfTI data')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='VoxelMorph',
                        choices=['VoxelMorph', 'NestedMorph'],
                        help='Model to train')
    
    # Data arguments
    parser.add_argument('--data-dirs', type=str, nargs='+', required=True,
                        help='List of training dataset directories')
    parser.add_argument('--data-name', type=str, default='mrct',
                        help='Dataset name (for output directory)')
    parser.add_argument('--img-size', type=int, nargs=3, default=[192, 192, 192],
                        help='Image size (H W D)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Train/validation split ratio')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    # System arguments
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device to use')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU")
    
    # Convert img_size to tuple
    args.img_size = tuple(args.img_size)
    
    # Validate data directories
    for data_dir in args.data_dirs:
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Start training
    train_model(args)


if __name__ == "__main__":
    main()

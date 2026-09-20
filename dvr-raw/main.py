import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader        
import torchio as tio
import collections
import random
import os
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
import argparse
import csv
import time
import gc
import joblib
import torch.distributed as dist

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
if alloc_conf:
    print(f"✅ PyTorch CUDA Allocator Conf is set to: {alloc_conf}")
else:
    print("❌ WARNING: PYTORCH_CUDA_ALLOC_CONF is NOT set.")

from sklearn.model_selection import train_test_split

from dvr2.dino_encoder import DINOEncoder
from dvr2.vlearn_rob import train_vlearn
from dvr2.test import evaluate_vlearn, precompute_features_in_memory
from dvr2.utils import prepare_train_dataset, prepare_test_dataset, get_augmentation_transform, vlearn_collate_fn_train, vlearn_collate_fn_val
from dvr2.extract_dino import extract_dino_features, train_dino_head
from dvr2.train_student import train_student_model
from dvr2.dataset import TestDataset, VlearnDataset, HeadDataset
from dvr2.student_extractor import StudentEncoder3D


if __name__ == '__main__':
    # Enable early memory monitoring
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear any existing cache
        memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"🔍 Initial GPU Memory Status: {memory_total:.2f}GB total")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training phase")
    parser.add_argument("--test", action="store_true", help="Run evaluation phase")
    parser.add_argument("--dino", action="store_true", help="Run DINOv2 feature encoding phase")
    parser.add_argument("--student", action="store_true", help="Run student model training phase")
    parser.add_argument("--data_name", type=str, default=None, help="Name of dataset")
    parser.add_argument("--model_name", type=str, default=None, help="Name of training dataset")
    parser.add_argument("--train_dir", type=str, default="Train", help="Path to training folder")
    parser.add_argument("--train_dir2", type=str, default="Train 2", help="Path to second training folder") 
    parser.add_argument("--model_dir", type=str, default="None", help="Path to saved model from training.")
    parser.add_argument("--test_dir", type=str, default="Test", help="Path to test folder")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=5, help="Number of deformation steps")
    parser.add_argument("--resume", action="store_true", help="Continue training from the last checkpoint")
    parser.add_argument("--dataset_type", type=str, default=None, help="Anatomy of dataset (brain or abdomen)")
    parser.add_argument('--use-raw-images', action='store_true', help='Use raw images as input instead of DINO features for ablation.')
    parser.add_argument("--dino_version", type=str, default="dinov2", choices=['dinov2', 'dinov3'], help="DINO model version to use for feature extraction")
    args = parser.parse_args()
    
    free, total = torch.cuda.mem_get_info()
    print(f"Free memory: {free/1024**3:.2f}GB, Total memory: {total/1024**3:.2f}GB")

    # device already defined above
    dino_encoder = DINOEncoder(device=device, dino_version=args.dino_version)
    model_name = args.model_name if args.model_name else args.data_name
    dino_dir = os.path.join(args.model_dir, "Dino-Features", f"{args.dino_version}", f"{model_name}")
    model_dir = os.path.join(args.model_dir, "Models", f"{args.dino_version}", f"{model_name}")
    
    if args.dataset_type is not None:
        dino_dir = os.path.join(args.model_dir, "Dino-Features", "Int-layer", f"{args.dino_version}", f"{model_name}")
        model_dir = os.path.join(args.model_dir, "Ablation", "Int-layer", f"{args.dino_version}", f"{model_name}")
    
    if args.use_raw_images:
        model_dir = os.path.join(args.model_dir, "Ablation", "Raw-img", f"{args.dino_version}", f"{model_name}")
            
    os.makedirs(dino_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if args.train:            
        train_paths = prepare_train_dataset(args.train_dir)
        if args.train_dir2 != "Train 2":
            train2_paths = prepare_train_dataset(args.train_dir2)
            train_paths += train2_paths
    
        # Create validation split from training data TODO: potentially change train/val split
        train_paths, val_paths = train_test_split(train_paths, test_size=0.15, random_state=42)
        print(f"Data split: {len(train_paths)} training samples, {len(val_paths)} validation samples.")

        AUGMENTATIONS_PER_IMAGE = 10
            
        # if args.dino:
            # Extract features
            # extract_dino_features(dino_encoder, train_paths, dino_dir, AUGMENTATIONS_PER_IMAGE, head_dataset=head_dataset, set_type="train")
            # extract_dino_features(dino_encoder, val_paths, dino_dir, AUGMENTATIONS_PER_IMAGE, set_type="val")
            
        # if args.student:
            # Train student model on DINO features
            # train_student_model(dino_dir, model_dir, device)

        train_dataset = VlearnDataset(
            path_pairs=train_paths,
            transform=get_augmentation_transform()
        )
            
        val_dataset = VlearnDataset(
            path_pairs=val_paths,
            transform=None
        )
            
        print("Inspecting first data sample to determine shapes...")
        # mr_vol, _ = train_dataset[0]
        # vol_shape = mr_vol.shape[1:]
        vol_shape = (128, 128, 128)  # Hardcoded volume shape for consistency
        feature_shape = (64, 128, 16, 16)  # Hardcoded feature shape for DINO
            
        # Load trained Student Feature Extractor ---
        # print("🧠 Loading trained student feature extractor...")
        # student_extractor = StudentEncoder3D(in_channels=1, out_channels=feature_shape[0]) # out_channels=64
        # student_extractor.load_state_dict(torch.load(os.path.join(model_dir, "student_feature_extractor.pth")))
        # student_extractor.to(device)
        # student_extractor.eval() # IMPORTANT: Set to evaluation mode
            
        head_dataset = HeadDataset(train_paths, transform=get_augmentation_transform())
        pca_transformer = train_dino_head(dino_encoder, train_paths, dino_dir, target_size=(128, 128, 128), head_dataset=head_dataset, set_type="train")

        # Pre-compute and cache the validation set in RAM
        val_image_loader = DataLoader(val_dataset, batch_size=1) # Batch size 1 for individual processing
        val_data_cached = precompute_features_in_memory(
            val_image_loader, dino_encoder, pca_transformer, device, use_raw_images=args.use_raw_images, dataset_type=args.dataset_type
        )
        
        feature_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=8,  # Try original batch size first
            shuffle=True,
            num_workers=0,  
            pin_memory=True,
            collate_fn=vlearn_collate_fn_train
        )
        
        val_loader = DataLoader(
            val_data_cached, 
            batch_size=4, 
            shuffle=False,
            num_workers=0,  
            pin_memory=True,
            collate_fn=vlearn_collate_fn_val
        )

        # Clear memory before training to avoid OOM
        print("\nClearing GPU memory before training...")
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Check memory after clearing
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            memory_free = torch.cuda.get_device_properties(device).total_memory / 1024**3 - memory_reserved
            print(f"GPU Memory after clearing: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved, {memory_free:.2f}GB free")

        print("\nStarting V-Learn training...")
        # Try small network first with memory optimizations
        # Fall back to smaller if OOM occurs

        train_vlearn(dino_encoder, pca_transformer, feature_loader, val_loader, device, epochs=args.epochs, max_steps=args.steps, save_dir=model_dir, dino_dir=dino_dir, is_continue_training=args.resume, use_raw_images=args.use_raw_images, dataset_type=args.dataset_type)
    
        def aggressive_memory_cleanup():
            """Aggressively clear all GPU memory"""
            import gc
            # Clear Python garbage
            gc.collect()
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
            # Force another garbage collection
            gc.collect()
        """   
        try:
            print("🎯 Attempting training with MEDIUM network for optimal performance...")
            train_vlearn(feature_loader, val_loader, device, vol_shape, feature_shape, epochs=args.epochs, max_steps=args.steps, save_dir=model_dir, network_size="medium")
        except torch.cuda.OutOfMemoryError as e:
            print(f"⚠️  Medium network OOM: {str(e)[:100]}...")
            print("🔄 Aggressively clearing memory and falling back to SMALL network...")

            # Clear everything from the failed attempt
            aggressive_memory_cleanup()
            time.sleep(2)  # Give the GPU a moment
            
            try:
                print("🎯 Attempting training with SMALL network for optimal performance...")
                train_vlearn(feature_loader, val_loader, device, vol_shape, feature_shape, epochs=args.epochs, max_steps=args.steps, save_dir=model_dir, network_size="small")
            except torch.cuda.OutOfMemoryError as e:
                print(f"⚠️  Small network OOM: {str(e)[:100]}...")
                print("🔄 Aggressively clearing memory and falling back to TINY network...")

                # Clear everything from the failed attempt
                aggressive_memory_cleanup()
                time.sleep(2)  # Give the GPU a moment

                print("🎯 Attempting training with TINY network for optimal performance...")
                train_vlearn(feature_loader, val_loader, device, vol_shape, feature_shape, epochs=args.epochs, max_steps=args.steps, save_dir=model_dir, network_size="tiny")
""" 
    if args.test:
        log_path = os.path.join(model_dir, f"{args.data_name}_test_results.csv")
        
        print("\nPreparing test data...")
        test_paths, label_paths = prepare_test_dataset(args.test_dir)
            
        save_path = os.path.join(model_dir, "best_proj_head.pth")
            
        print("\nCreating test dataset object...")
        test_dataset = TestDataset(
            vol_pairs=test_paths,
            label_pairs=label_paths,
        )
        
        print("Inspecting first data sample to determine shapes...")
        mr_vol_sample, _, _, _, _ = test_dataset[0]
        vol_shape = mr_vol_sample.shape[1:]
        feature_shape = (64, mr_vol_sample.shape[1], 16, 16)  # Hardcoded feature shape for DINO
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=16, 
            shuffle=False,
            num_workers=4,  
            pin_memory=True,
            collate_fn=vlearn_collate_fn_val
        )
        
        test_folder = os.path.join(model_dir, f"test-{args.data_name}")
        os.makedirs(test_folder, exist_ok=True)
        
        proj_head_path = os.path.join(dino_dir, "best_proj_head.pth")
        print(f"Loading pre-trained projection head from {proj_head_path}...")
        dino_encoder.load(proj_head_path)
        dino_encoder.freeze_dino()
        # Also freeze the projection head, as it's already trained
        for param in dino_encoder.proj_head.parameters():
            param.requires_grad = False
        
        # Load the PCA model fitted on the training data
        pca_path = os.path.join(dino_dir, "pca_transformer.pkl") 
        pca_transformer = joblib.load(pca_path)
        
        print("\nRunning V-Learn evaluation...")
        metrics = evaluate_vlearn(test_loader, test_folder, vol_shape, model_dir, device, dino_encoder, pca_transformer, args.use_raw_images, max_steps=5, dataset_type=args.dataset_type)

        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
                
        # Write all metrics to CSV file
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in metrics.items():
                writer.writerow([k, v])

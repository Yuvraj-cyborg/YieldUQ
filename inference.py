#!/usr/bin/env python3
"""
Quick inference script - loads trained model and runs uncertainty estimation
without retraining or hyperparameter search.

Usage:
    python inference.py --checkpoint checkpoints/model_epoch_98_r2_0.9534.pt
"""

import torch
from torch_geometric.loader import DataLoader
from src.dataset import ReactionDataset
from src.model import GNNModel
from src.train import mc_dropout_predict, heteroscedastic_predict
from src.evaluate import analyze_uncertainty, check_coverage
from src.utils import set_seed
from sklearn.model_selection import train_test_split
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='YieldUQ-GNN: Inference Only')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--mc_samples', type=int, default=100, help='MC Dropout samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        if os.path.exists('checkpoints'):
            for f in sorted(os.listdir('checkpoints')):
                if f.endswith('.pt'):
                    print(f"  - checkpoints/{f}")
        return
    
    print("\n" + "="*70)
    print("YIELDIQ-GNN: INFERENCE MODE (No Training)")
    print("="*70)
    print(f"\nLoading checkpoint: {args.checkpoint}")
    
    # Load checkpoint to inspect configuration
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = ReactionDataset("data/Dreher_and_Doyle_input_data.xlsx")
    category_sizes = {k: len(v) for k, v in dataset.category_maps.items()}
    print(f"Dataset loaded: {len(dataset)} reactions")
    
    # Split dataset (same split as training)
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=args.seed)
    test_dataset = dataset[test_idx]
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Infer model configuration from checkpoint
    state_dict = checkpoint['model_state_dict']
    
    # Detect hidden_channels from first conv layer
    first_conv_weight_key = 'convs.0.nn.0.weight'
    if first_conv_weight_key in state_dict:
        hidden_channels = state_dict[first_conv_weight_key].shape[0]
    else:
        hidden_channels = 128  # fallback
    
    # Detect num_layers from number of conv layers
    num_layers = sum(1 for k in state_dict.keys() if k.startswith('convs.') and '.nn.0.weight' in k)
    
    # Detect pooling type
    if 'pool.gate_nn.0.weight' in state_dict:
        pooling = 'attention'
    elif any('pool.lstm' in k for k in state_dict.keys()):
        pooling = 'set2set'
    else:
        pooling = 'mean'
    
    # Detect heteroscedastic
    heteroscedastic = 'lin_logvar.weight' in state_dict
    
    # Detect embed_dim
    embed_key = 'emb_ligand.weight'
    if embed_key in state_dict:
        embed_dim = state_dict[embed_key].shape[1]
    else:
        embed_dim = 16
    
    print(f"\nDetected model configuration:")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Num layers: {num_layers}")
    print(f"  Pooling: {pooling}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Heteroscedastic: {heteroscedastic}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  R² Score: {checkpoint.get('r2', 'unknown'):.4f}")
    
    # Create model with detected configuration
    model = GNNModel(
        in_channels=6,
        edge_dim=4,
        hidden_channels=hidden_channels,
        category_sizes=category_sizes,
        embed_dim=embed_dim,
        pooling=pooling,
        use_edge_features=True,
        heteroscedastic=heteroscedastic,
        num_layers=num_layers
    )
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    print("\n✅ Model loaded successfully!")
    
    # Collect true labels
    y_true = []
    for batch in test_loader:
        y_true += batch.y.flatten().tolist()
    
    # Uncertainty estimation
    print("\n" + "="*70)
    print("UNCERTAINTY ESTIMATION")
    print("="*70)
    
    # 1. Heteroscedastic (if available)
    if heteroscedastic:
        print("\n1. HETEROSCEDASTIC UNCERTAINTY (Fast)")
        print("-" * 70)
        means, stds = heteroscedastic_predict(model, test_loader)
        results_hetero = analyze_uncertainty((means, stds), y_true, method='heteroscedastic')
        check_coverage(y_true, means.numpy(), stds.numpy())
    
    # 2. MC Dropout
    print(f"\n2. MONTE CARLO DROPOUT ({args.mc_samples} samples)")
    print("-" * 70)
    mc_preds = mc_dropout_predict(model, test_loader, n_samples=args.mc_samples)
    results_mc = analyze_uncertainty(mc_preds, y_true, method='mc_dropout')
    
    # Visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    from src.visualize import visualize_all_results, compare_uncertainty_methods
    
    if heteroscedastic:
        print("\nShowing Heteroscedastic results...")
        visualize_all_results(y_true, (results_hetero['mean_pred'], results_hetero['std_pred']),
                            title_suffix=" (Heteroscedastic)")
    
    print("\nShowing MC Dropout results...")
    visualize_all_results(y_true, mc_preds, title_suffix=" (MC Dropout)")
    
    # Compare methods
    if heteroscedastic:
        print("\nComparing uncertainty methods...")
        compare_uncertainty_methods(
            y_true,
            [results_hetero, results_mc],
            ['Heteroscedastic', 'MC Dropout']
        )
    
    print("\n" + "="*70)
    print("✅ INFERENCE COMPLETE!")
    print("="*70)
    print("\nYour trained model achieved:")
    print(f"  R² Score: {results_mc['r2']:.4f}")
    print(f"  MAE: {results_mc['mae']:.4f}")
    print(f"  RMSE: {results_mc['rmse']:.4f}")
    print(f"  Uncertainty-Error Correlation: {results_mc['correlation']:.4f}")
    print(f"  Expected Calibration Error: {results_mc['ece']:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

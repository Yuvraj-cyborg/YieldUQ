import torch
from torch_geometric.loader import DataLoader
from src.dataset import ReactionDataset
from src.model import GNNModel, EnsembleModel
from src.train import (train_epoch, evaluate, mc_dropout_predict, 
                       heteroscedastic_predict, ensemble_predict,
                       save_checkpoint, load_checkpoint)
from src.evaluate import analyze_uncertainty, check_coverage
from src.utils import set_seed
from sklearn.model_selection import train_test_split
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='YieldUQ-GNN: Enhanced Training')
    
    # Model architecture
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--embed_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--pooling', type=str, default='attention', 
                       choices=['mean', 'attention', 'set2set'], help='Pooling method')
    parser.add_argument('--use_edge_features', action='store_true', default=True,
                       help='Use edge features with GINEConv')
    parser.add_argument('--heteroscedastic', action='store_true', default=True,
                       help='Use heteroscedastic uncertainty')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    
    # Uncertainty
    parser.add_argument('--mc_samples', type=int, default=100, help='MC Dropout samples')
    parser.add_argument('--use_ensemble', action='store_true', default=False,
                       help='Train ensemble of models')
    parser.add_argument('--num_ensemble', type=int, default=5, help='Ensemble size')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to checkpoint')
    
    return parser.parse_args()


def hyperparameter_search(dataset, category_sizes):
    """Grid search for optimal hyperparameters."""
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH")
    print("="*60 + "\n")
    
    # Define search space
    hidden_channels_options = [64, 128, 256]
    num_layers_options = [2, 3, 4]
    pooling_options = ['mean', 'attention', 'set2set']
    
    best_r2 = -float('inf')
    best_config = None
    results = []
    
    for hidden in hidden_channels_options:
        for num_layers in num_layers_options:
            for pooling in pooling_options:
                print(f"\nTesting: hidden={hidden}, layers={num_layers}, pooling={pooling}")
                
                # Split dataset
                train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
                train_dataset = dataset[train_idx]
                test_dataset = dataset[test_idx]
                
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
                
                # Create model
                model = GNNModel(
                    in_channels=6,
                    edge_dim=4,
                    hidden_channels=hidden,
                    category_sizes=category_sizes,
                    embed_dim=16,
                    pooling=pooling,
                    use_edge_features=True,
                    heteroscedastic=True,
                    num_layers=num_layers
                )
                
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=10
                )
                
                # Quick training (20 epochs for search)
                for epoch in range(20):
                    train_epoch(model, train_loader, optimizer, use_heteroscedastic=True)
                    r2 = evaluate(model, test_loader, use_heteroscedastic=True)
                    scheduler.step(r2)
                
                final_r2 = evaluate(model, test_loader, use_heteroscedastic=True)
                print(f"  Final R²: {final_r2:.4f}")
                
                results.append({
                    'hidden': hidden,
                    'num_layers': num_layers,
                    'pooling': pooling,
                    'r2': final_r2
                })
                
                if final_r2 > best_r2:
                    best_r2 = final_r2
                    best_config = {
                        'hidden_channels': hidden,
                        'num_layers': num_layers,
                        'pooling': pooling
                    }
    
    print("\n" + "="*60)
    print("SEARCH RESULTS")
    print("="*60)
    for res in sorted(results, key=lambda x: x['r2'], reverse=True):
        print(f"R²={res['r2']:.4f} | hidden={res['hidden']}, layers={res['num_layers']}, pooling={res['pooling']}")
    
    print(f"\nBest Configuration: {best_config}")
    print(f"Best R²: {best_r2:.4f}")
    print("="*60 + "\n")
    
    return best_config


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("\n" + "="*60)
    print("YIELDIQ-GNN: ENHANCED TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Hidden Channels: {args.hidden_channels}")
    print(f"  Num Layers: {args.num_layers}")
    print(f"  Pooling: {args.pooling}")
    print(f"  Edge Features: {args.use_edge_features}")
    print(f"  Heteroscedastic: {args.heteroscedastic}")
    print(f"  Ensemble: {args.use_ensemble} ({'x'+str(args.num_ensemble) if args.use_ensemble else ''})")
    print(f"  MC Samples: {args.mc_samples}")
    print("="*60 + "\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = ReactionDataset("data/Dreher_and_Doyle_input_data.xlsx")
    category_sizes = {k: len(v) for k, v in dataset.category_maps.items()}
    print(f"Dataset loaded: {len(dataset)} reactions")
    print(f"Category sizes: {category_sizes}\n")
    
    # Optional: Run hyperparameter search
    if input("Run hyperparameter search? (y/n): ").lower() == 'y':
        best_config = hyperparameter_search(dataset, category_sizes)
        args.hidden_channels = best_config['hidden_channels']
        args.num_layers = best_config['num_layers']
        args.pooling = best_config['pooling']
    
    # Split dataset
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=args.seed)
    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    if args.use_ensemble:
        print(f"Creating ensemble of {args.num_ensemble} models...")
        model = EnsembleModel(
            num_models=args.num_ensemble,
            in_channels=6,
            edge_dim=4,
            hidden_channels=args.hidden_channels,
            category_sizes=category_sizes,
            embed_dim=args.embed_dim,
            pooling=args.pooling,
            use_edge_features=args.use_edge_features,
            heteroscedastic=args.heteroscedastic,
            num_layers=args.num_layers
        )
    else:
        model = GNNModel(
            in_channels=6,
            edge_dim=4,
            hidden_channels=args.hidden_channels,
            category_sizes=category_sizes,
            embed_dim=args.embed_dim,
            pooling=args.pooling,
            use_edge_features=args.use_edge_features,
            heteroscedastic=args.heteroscedastic,
            num_layers=args.num_layers
        )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15
    )
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, args.load_checkpoint)
    
    # Training loop
    print("\nStarting training...")
    best_r2 = -float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        loss = train_epoch(model, train_loader, optimizer, use_heteroscedastic=args.heteroscedastic)
        r2 = evaluate(model, test_loader, use_heteroscedastic=args.heteroscedastic)
        scheduler.step(r2)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f} | R²: {r2:.4f}")
        
        # Save checkpoint
        if r2 > best_r2:
            best_r2 = r2
            save_checkpoint(model, optimizer, epoch+1, r2, path='checkpoints')
        
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch+1, r2, path='checkpoints')
    
    print(f"\nTraining complete! Best R²: {best_r2:.4f}")
    
    # Collect true labels
    y_true = []
    for batch in test_loader:
        y_true += batch.y.flatten().tolist()
    
    # Uncertainty estimation comparisons
    print("\n" + "="*60)
    print("UNCERTAINTY ESTIMATION COMPARISON")
    print("="*60)
    
    # 1. Heteroscedastic uncertainty
    if args.heteroscedastic:
        print("\n1. HETEROSCEDASTIC UNCERTAINTY (Single Forward Pass)")
        print("-" * 60)
        means, stds = heteroscedastic_predict(model, test_loader)
        results_hetero = analyze_uncertainty((means, stds), y_true, method='heteroscedastic')
        check_coverage(y_true, means.numpy(), stds.numpy())
    
    # 2. MC Dropout
    print("\n2. MONTE CARLO DROPOUT (100 Forward Passes)")
    print("-" * 60)
    mc_preds = mc_dropout_predict(model, test_loader, n_samples=args.mc_samples)
    results_mc = analyze_uncertainty(mc_preds, y_true, method='mc_dropout')
    
    # 3. Ensemble (if used)
    if args.use_ensemble:
        print("\n3. ENSEMBLE UNCERTAINTY")
        print("-" * 60)
        ens_means, ens_stds = ensemble_predict(model, test_loader)
        results_ens = analyze_uncertainty((ens_means, ens_stds), y_true, method='ensemble')
        check_coverage(y_true, ens_means.numpy(), ens_stds.numpy())
    
    # Visualize all results
    from src.visualize import visualize_all_results, compare_uncertainty_methods
    
    print("\nGenerating visualizations...")
    
    if args.heteroscedastic:
        visualize_all_results(y_true, (results_hetero['mean_pred'], results_hetero['std_pred']),
                            title_suffix="(Heteroscedastic)")
    
    visualize_all_results(y_true, mc_preds, title_suffix="(MC Dropout)")
    
    # Compare methods
    if args.heteroscedastic:
        compare_uncertainty_methods(
            y_true,
            [results_hetero, results_mc],
            ['Heteroscedastic', 'MC Dropout']
        )
    
    print("\nAll done! Check the plots directory for visualizations.")


if __name__ == "__main__":
    main()

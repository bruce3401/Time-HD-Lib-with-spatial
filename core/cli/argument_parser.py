"""
Command Line Interface - Argument Parser

This module provides comprehensive command line argument parsing functionality
for the High-Dimensional Time Series Analysis Framework.
"""

import argparse
from typing import Dict, Any


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command line argument parser.
    
    Returns:
        Configured ArgumentParser instance with all framework options
    """
    parser = argparse.ArgumentParser(
        description='High-Dimensional Time Series Analysis Framework', 
        argument_default=argparse.SUPPRESS,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Basic configuration
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='Task type: [long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, 
                        help='Training mode: 1 for training, 0 for testing only')
    parser.add_argument('--model', type=str, required=True, default='UCast',
                        help='Model name (see --list-models for available options)')

    # Data loading configuration
    parser.add_argument('--data', type=str, required=True, default='ETTh1', 
                        help='Dataset name')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', 
                        help='Root directory path for dataset files')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', 
                        help='Specific data file name')
    parser.add_argument('--features', type=str, default='M',
                        help='Forecasting mode: [M]ultivariate, [S]ingle variable, [MS]ultivariate to single')
    parser.add_argument('--target', type=str, default='OT', 
                        help='Target feature name for S or MS tasks')
    parser.add_argument('--freq', type=str, default='h',
                        help='Time frequency: [s]econdly, [t]minutely, [h]ourly, [d]aily, [b]usiness days, [w]eekly, [m]onthly')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', 
                        help='Directory for saving model checkpoints')

    # Forecasting task parameters
    parser.add_argument('--seq_len', type=int, default=None, 
                        help='Input sequence length (lookback window)')
    parser.add_argument('--label_len', type=int, default=0, 
                        help='Start token length for decoder models')
    parser.add_argument('--pred_len', type=int, default=None,
                        help='Prediction sequence length (forecast horizon)')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', 
                        help='Seasonal patterns for M4 dataset')
    parser.add_argument('--inverse', action='store_true', default=False,
                        help='Apply inverse transformation to denormalize outputs')
    parser.add_argument('--seq_len_factor', type=int, default=4, 
                        help='Multiplier for automatic sequence length calculation')

    # Model architecture parameters
    parser.add_argument('--expand', type=int, default=2, 
                        help='Expansion factor for Mamba-based models')
    parser.add_argument('--d_conv', type=int, default=4, 
                        help='Convolution kernel size for Mamba models')
    parser.add_argument('--top_k', type=int, default=5, 
                        help='Top-k parameter for TimesNet model')
    parser.add_argument('--num_kernels', type=int, default=6, 
                        help='Number of kernels for Inception blocks')
    parser.add_argument('--enc_in', type=int, default=7, 
                        help='Number of input channels/features')
    parser.add_argument('--dec_in', type=int, default=7, 
                        help='Number of decoder input channels')
    parser.add_argument('--c_out', type=int, default=7, 
                        help='Number of output channels/features')
    parser.add_argument('--d_model', type=int, default=512, 
                        help='Model embedding dimension')
    parser.add_argument('--n_heads', type=int, default=8, 
                        help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, 
                        help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, 
                        help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, 
                        help='Feedforward network dimension')
    parser.add_argument('--moving_avg', type=int, default=25, 
                        help='Moving average window size for decomposition')
    parser.add_argument('--factor', type=int, default=1, 
                        help='Attention factor for ProbSparse attention')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropout rate for regularization')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Time features encoding: [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', 
                        help='Activation function')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='Channel processing: 0=dependent, 1=independent (for FreTS)')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='Series decomposition method: [moving_avg, dft_decomp]')
    parser.add_argument('--use_norm', type=int, default=1, 
                        help='Apply normalization: 1=True, 0=False')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Alpha parameter for specific models')
    parser.add_argument('--channel_reduction_ratio', type=float, default=16,
                        help='Channel reduction ratio for attention mechanisms')

    # SpatialLCA-specific parameters
    parser.add_argument('--r_star', type=int, default=16,
                        help='SpatialLCA: number of latent centroids per head')
    parser.add_argument('--gumbel_alpha', type=float, default=0.15,
                        help='SpatialLCA: Gumbel-softmax noise weight for centroid assignment')
    parser.add_argument('--patch_size', type=int, default=14,
                        help='SpatialLCA / patch-based models: patch size')
    parser.add_argument('--patch_stride', type=int, default=7,
                        help='SpatialLCA / patch-based models: patch stride')
    parser.add_argument('--use_time_enc', action='store_true', default=False,
                        help='SpatialLCA: prepend a normalized time encoding to each patch')
    parser.add_argument('--keep_patches', action='store_true', default=False,
                        help='SpatialLCA: keep per-patch tokens instead of pooling them')
    parser.add_argument('--attn', type=str, default='lca',
                        help='SpatialLCA: attention variant [lca, full, linformer]')
    parser.add_argument('--backbone', type=str, default='lca',
                        help='SpatialLCA: backbone [lca, mixer]')
    parser.add_argument('--linformer_k', type=int, default=64,
                        help='SpatialLCA: Linformer rank when attn=linformer')
    parser.add_argument('--use_mixer_aux', action='store_true', default=False,
                        help='SpatialLCA: add MLP-Mixer auxiliary residual branch')
    parser.add_argument('--mixer_hidden', type=int, default=256,
                        help='SpatialLCA: MLP-Mixer hidden dim')
    parser.add_argument('--mixer_blocks', type=int, default=2,
                        help='SpatialLCA: MLP-Mixer block count')
    parser.add_argument('--use_revin', action='store_true', default=False,
                        help='SpatialLCA: enable RevIN normalization')
    parser.add_argument('--revin_no_affine', action='store_true', default=False,
                        help='SpatialLCA: disable RevIN affine parameters')
    parser.add_argument('--use_dishts', action='store_true', default=False,
                        help='SpatialLCA: enable DishTS normalization')
    parser.add_argument('--subtract_last', action='store_true', default=False,
                        help='SpatialLCA: subtract last value normalization')
    parser.add_argument('--use_decomp', action='store_true', default=False,
                        help='SpatialLCA: enable trend/seasonal decomposition')
    parser.add_argument('--decomp_kernel', type=int, default=25,
                        help='SpatialLCA: moving-average kernel for decomposition')
    parser.add_argument('--mlp_head', action='store_true', default=False,
                        help='SpatialLCA: use MLP prediction head')
    parser.add_argument('--indep_head', action='store_true', default=False,
                        help='SpatialLCA: per-channel independent prediction head')
    parser.add_argument('--head_hidden_ratio', type=int, default=1,
                        help='SpatialLCA: MLP head hidden ratio')
    parser.add_argument('--grad_ckpt', action='store_true', default=False,
                        help='SpatialLCA: enable gradient checkpointing')
    parser.add_argument('--channel_chunk', type=int, default=0,
                        help='SpatialLCA: split channels into chunks at the patch embedder (0 disables)')
    parser.add_argument('--use_laplacian_smooth', action='store_true', default=False,
                        help='SpatialLCA: enable Laplacian smoothing on centroid assignment (Tobler-style)')
    parser.add_argument('--laplacian_lambda', type=float, default=0.01,
                        help='SpatialLCA: weight on Laplacian smoothness regulariser')
    parser.add_argument('--use_gcn_branch', action='store_true', default=False,
                        help='SpatialLCA E5: parallel GCN branch fused at encoder output via sigmoid gate')
    parser.add_argument('--gcn_layers', type=int, default=2,
                        help='SpatialLCA E5: number of GCN layers in the parallel branch')
    parser.add_argument('--diffusion_K', type=int, default=2,
                        help='DCRNN: number of diffusion steps per DConv (canonical=2)')
    parser.add_argument('--report_raw_metrics', action='store_true', default=False,
                        help='Report raw-flow (inverse-transformed) test metrics in addition to z-score MSE')
    parser.add_argument('--mask_value', type=float, default=1e-3,
                        help='Zero-mask threshold for raw-flow metric reporting')
    parser.add_argument('--heat_weeks_csv', type=str, default=None,
                        help='Mobility-CA case study: path to NOAA heat-week CSV (e.g. dataset/NOAA/heat_weeks_2018_2020.csv). Enables flag=heat split.')
    parser.add_argument('--exclude_heat_from_train', action='store_true', default=False,
                        help='Mobility-CA case study: skip heat-event weeks from training set (must specify --heat_weeks_csv)')

    # Training optimization parameters
    parser.add_argument('--num_workers', type=int, default=2, 
                        help='Number of data loader worker processes')
    parser.add_argument('--itr', type=int, default=1, 
                        help='Number of experiment iterations')
    parser.add_argument('--train_epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Training batch size')
    parser.add_argument('--patience', type=int, default=3, 
                        help='Early stopping patience (epochs)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (AdamW). >0 switches optimizer Adam→AdamW')
    parser.add_argument('--scale_mode', type=str, default='soft',
                        choices=['soft', 'topk', 'grouped'],
                        help='RegionFormer Step 2 mode: soft=O(N²) softmask, topk=hard top-K within region, grouped=CA fallback (NYI)')
    parser.add_argument('--topk_within', type=int, default=8,
                        help='When scale_mode=topk, K nearest co-region sensors per i')
    parser.add_argument('--ablate_step', type=str, default=None,
                        choices=[None, 'within', 'cross'],
                        help='RegionFormer ablation: "within" zeros A_wr (cross-region only ≡ classic LCA), '
                             '"cross" zeros bcast(A_cr) (within-region only). Default None = full model.')
    parser.add_argument('--use_coord_embed', action='store_true', default=False,
                        help='RegionFormer: inject Fourier-feature lat/lon coordinate embeddings into '
                             'the variate tokens (NeRF-style positional encoding in 2D space). '
                             'Information iTransformer cannot use — variates are not exchangeable in physical space.')
    parser.add_argument('--coord_freqs', type=int, default=8,
                        help='Number of Fourier frequencies for --use_coord_embed (per axis). Default 8 → 32-D feature.')
    parser.add_argument('--use_dense_attn', action='store_true', default=False,
                        help='RegionFormer: add a parallel iTransformer-style dense cross-variate '
                             'attention path with a learnable scalar gate (init 0). In the limit '
                             'γ→1, RF subsumes iTransformer, so RF >= iTr by construction.')
    parser.add_argument('--hard_assignment', action='store_true', default=False,
                        help='RegionFormer §6.2 ablation: replace soft W with onehot(argmax(W)) in the '
                             'forward pass via straight-through estimator. Isolates the value of soft '
                             'probabilistic mixing while keeping training viable.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'lion', 'adafactor'],
                        help='Optimizer choice. lion = Google 2023 sign-update optimizer; often 1-3% better on time-series.')
    parser.add_argument('--use_adaptive_adj', action='store_true', default=False,
                        help='RegionFormer: low-rank learnable adjacency residual blended with fixed A_hat. '
                             'Requires --use_graph_prop. Targets channels where Queen/KNN adjacency is wrong proxy '
                             'for relevance (esp. mobility — socioeconomic >> geographic neighbours).')
    parser.add_argument('--adaptive_adj_dim', type=int, default=8,
                        help='Embedding dim K for --use_adaptive_adj. 2NK total params; default K=8.')
    parser.add_argument('--use_timemixup', action='store_true', default=False,
                        help='TimeMixUp augmentation: convex combine two batch samples with λ~Beta(α,α). '
                             'Standard 2024 time-series augmentation; targets low-data channels (Mobility).')
    parser.add_argument('--timemixup_alpha', type=float, default=0.2,
                        help='Beta distribution α for --use_timemixup; smaller=closer to original (less aggressive).')
    parser.add_argument('--input_mask_ratio', type=float, default=0.0,
                        help='Random temporal masking: zero out this fraction of input time-steps per sample (training only). '
                             'Robustness to missing observations; helps low-data regimes. 0.1-0.2 typical.')
    parser.add_argument('--des', type=str, default='Exp', 
                        help='Experiment description')
    parser.add_argument('--loss', type=str, default='MSE', 
                        help='Loss function')
    parser.add_argument('--lradj', type=str, default='type1', 
                        help='Learning rate adjustment strategy')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Enable automatic mixed precision training')

    # GPU and distributed training
    parser.add_argument('--use_gpu', type=bool, default=True, 
                        help='Enable GPU acceleration')
    parser.add_argument('--gpu', type=str, default=None, 
                        help='GPU device ID or comma-separated list (e.g., "0" or "0,2,3,7")')



    # Hyperparameter search
    parser.add_argument('--hyper_parameter_searching', action='store_true', default=False,
                        help='Enable automated hyperparameter search')
    parser.add_argument('--hp_log_dir', type=str, default='./hp_logs/', 
                        help='Directory for hyperparameter search logs')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=2021, 
                        help='Random seed for reproducibility')

    return parser 
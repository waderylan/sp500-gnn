# Development mode — set to None to use the full universe
DEV_UNIVERSE_SIZE = None

# Reproducibility
RANDOM_SEED = 22

# Data download date range
DATA_START = "2015-01-01"
DATA_END   = "2025-12-31"

# Train / val / test split boundaries (inclusive end dates)
TRAIN_END = "2022-12-31"
VAL_END   = "2023-12-31"
TEST_END  = "2025-12-31"

# Universe filtering
MIN_COVERAGE = 0.95                  # fraction of trading days a stock must have data for

# Feature normalization
WINSORIZE_CLIP = (0.01, 0.99)        # cross-sectional winsorization percentiles
# With DEV_UNIVERSE_SIZE=50 the 1%/99% clip bites <1 stock per tail, so post-z-score
# extremes can reach ~6. With the full universe (~462 stocks) this can be tightened to 5.0.
NORM_MAX_ABS         = 6.0
MAX_FEATURE_NAN_FRAC = 0.05  # maximum allowed NaN fraction in the feature tensor

# Portfolio construction
MAX_WEIGHT           = 0.05          # maximum single-stock weight after normalization
TRANSACTION_COST_BPS = 10            # one-way cost in basis points
LONG_SHORT_QUANTILE  = 0.25          # fraction of stocks in each leg of the long-short portfolio
VOL_TARGET           = 0.10          # annualized target portfolio volatility for vol-targeted construction (10%)

# Graph construction — correlation
CORR_THRESHOLD    = 0.3             # edge threshold θ — set to ablation winner (see corr_threshold_ablation.json)
CORR_LOOKBACK_DAYS = 252            # rolling lookback window for correlation graph
CORR_ABLATION_THRESHOLDS = [0.3, 0.5, 0.7]  # θ values tested in the ablation study

# Three sample dates for correlation graph visualization (all in train or val)
CORR_SAMPLE_DATES = {
    "calm":   "2017-06-30",   # low-vol summer 2017 (train)
    "covid":  "2020-03-13",   # last Friday before worst COVID week (train)
    "recent": "2023-06-30",   # mid-val period (val)
}

# Graph construction — Granger
GRANGER_LAG        = 5              # number of lags for Granger causality F-test
GRANGER_CORRECTION = "bonferroni"   # "bonferroni" or "bh"; updated after ablation
GRANGER_MIN_EDGES  = 500            # fall back to BH if Bonferroni yields fewer edges

# GNN architecture
SAGE_FLOW  = "source_to_target"     # SAGEConv flow direction; critical for Granger directionality
HIDDEN_DIM     = 128   # updated from 64 after hparam search (04b_gnn_hparam_search.ipynb)
GNN_NUM_LAYERS = 3     # updated from 2 after hparam search
DROPOUT        = 0.3

# Training
LEARNING_RATE     = 0.001   # used by LSTM
GNN_LEARNING_RATE = 3e-4    # used by GNN; updated after hparam search
EARLY_STOP_PATIENCE     = 10        # epochs without val improvement before stopping
CHECKPOINT_EVERY_N_EPOCHS = 5

# LSTM
LSTM_HIDDEN_DIM = 64   # LSTM hidden units; not included in hparam search
LSTM_SEQ_LEN    = 4   # weeks of feature history per input sequence
LSTM_MAX_EPOCHS = 150  # upper bound; early stopping triggers first
GNN_MAX_EPOCHS  = 150  # upper bound; early stopping triggers first

# GNN hyperparameter grid search
# Fixed at CORR_THRESHOLD=0.3 (ablation winner). 24 configs total.
# Changes from round 1 (48 configs):
#   - num_layers fixed at 3 (won all top-5 slots; 2-layer configs dropped)
#   - hidden_dim shifted from [64, 128] to [128, 256] (64 never cracked top 5)
#   - lr shifted from [3e-4, 1e-3, 3e-3] to [1e-4, 3e-4, 1e-3] (winner was at low end; 3e-3 dropped)
#   - batch_norm kept [True, False]; True now means GraphNorm instead of BatchNorm1d
GNN_HPARAM_LR         = [1e-4, 3e-4, 1e-3]
GNN_HPARAM_HIDDEN     = [128, 256]
GNN_HPARAM_DROPOUT    = [0.1, 0.3]
GNN_HPARAM_BATCH_NORM = [True, False]  # True now means GraphNorm (was BatchNorm1d)
GNN_HPARAM_NUM_LAYERS = [3]            # fixed; 2-layer configs lost in round 1
GNN_HPARAM_PATIENCE   = 7  # reduced from EARLY_STOP_PATIENCE for faster search

# Paths — absolute, anchored to the directory containing this file
from pathlib import Path as _Path
_ROOT = _Path(__file__).parent

DATA_RAW_DIR      = str(_ROOT / "data/raw")
DATA_FEATURES_DIR = str(_ROOT / "data/features")
DATA_GRAPHS_DIR   = str(_ROOT / "data/graphs")
CORR_EDGES_DIR    = str(_ROOT / "data/graphs/corr_edges")
DATA_RESULTS_DIR  = str(_ROOT / "data/results")
CHECKPOINTS_DIR   = str(_ROOT / "data/results/checkpoints")
FIGURES_DIR       = str(_ROOT / "data/results/figures")

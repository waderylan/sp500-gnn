# Development mode — set to None to use the full universe
DEV_UNIVERSE_SIZE = 50

# Reproducibility
RANDOM_SEED = 22

# Data download date range
DATA_START = "2015-01-01"
DATA_END   = "2025-12-31"

# Universe construction filters
UNIVERSE_ADDED_BEFORE       = "2016-01-01"   # exclude stocks added on or after this date
UNIVERSE_NOT_REMOVED_BEFORE = "2024-12-01"   # exclude stocks removed before this date

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
NORM_MAX_ABS   = 6.0

# Portfolio construction
MAX_WEIGHT           = 0.05          # maximum single-stock weight after normalization
TRANSACTION_COST_BPS = 10            # one-way cost in basis points

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
HIDDEN_DIM = 64
DROPOUT    = 0.3

# Training
LEARNING_RATE           = 0.001
EARLY_STOP_PATIENCE     = 10        # epochs without val improvement before stopping
CHECKPOINT_EVERY_N_EPOCHS = 5

# LSTM
LSTM_SEQ_LEN    = 4   # weeks of feature history per input sequence
LSTM_MAX_EPOCHS = 150  # upper bound; early stopping triggers first
GNN_MAX_EPOCHS  = 150  # upper bound; early stopping triggers first

# Paths — absolute, anchored to the directory containing this file
from pathlib import Path as _Path
_ROOT = _Path(__file__).parent

DATA_RAW_DIR      = str(_ROOT / "data/raw")
DATA_FEATURES_DIR = str(_ROOT / "data/features")
DATA_GRAPHS_DIR   = str(_ROOT / "data/graphs")
DATA_RESULTS_DIR  = str(_ROOT / "data/results")
CHECKPOINTS_DIR   = str(_ROOT / "data/results/checkpoints")
FIGURES_DIR       = str(_ROOT / "data/results/figures")

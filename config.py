# Development mode — set to None to use the full universe
DEV_UNIVERSE_SIZE = 50

# Reproducibility
RANDOM_SEED = 22

# Train / val / test split boundaries (inclusive end dates)
TRAIN_END = "2022-12-31"
VAL_END   = "2023-12-31"
TEST_END  = "2025-12-31"

# Universe filtering
MIN_COVERAGE = 0.95                  # fraction of trading days a stock must have data for

# Feature normalization
WINSORIZE_CLIP = (0.01, 0.99)        # cross-sectional winsorization percentiles

# Portfolio construction
MAX_WEIGHT           = 0.05          # maximum single-stock weight after normalization
TRANSACTION_COST_BPS = 10            # one-way cost in basis points

# Graph construction — correlation
CORR_THRESHOLD    = 0.5             # edge threshold θ (tuned on val; ablation: {0.3, 0.5, 0.7})
CORR_LOOKBACK_DAYS = 252            # rolling lookback window for correlation graph

# Graph construction — Granger
GRANGER_LAG        = 5              # number of lags for Granger causality F-test
GRANGER_CORRECTION = "bonferroni"   # "bonferroni" or "bh"; updated after ablation

# GNN architecture
SAGE_FLOW  = "source_to_target"     # SAGEConv flow direction; critical for Granger directionality
HIDDEN_DIM = 64
DROPOUT    = 0.3

# Training
LEARNING_RATE           = 0.001
EARLY_STOP_PATIENCE     = 10        # epochs without val improvement before stopping
CHECKPOINT_EVERY_N_EPOCHS = 5

# Paths (relative to project root)
DATA_RAW_DIR      = "data/raw"
DATA_FEATURES_DIR = "data/features"
DATA_GRAPHS_DIR   = "data/graphs"
DATA_RESULTS_DIR  = "data/results"
CHECKPOINTS_DIR   = "data/results/checkpoints"
FIGURES_DIR       = "data/results/figures"

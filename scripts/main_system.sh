#!/bin/bash

#############################################
# Smart Home Security - Complete System
# XGBoost with ~90% Accuracy (More Realistic)
#############################################

export MAGICK_CONFIGURE_PATH=/dev/null 2>/dev/null
unset DISPLAY 2>/dev/null

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${MAGENTA}================================================================"
echo -e "  Smart Home Security - Realistic Detection System"
echo -e "  XGBoost with ~90% Accuracy Target"
echo -e "================================================================${NC}"
echo ""

NS3_DIR="$HOME/ns-allinone-3.43/ns-3.43"
SCRATCH_DIR="$NS3_DIR/scratch"
CURRENT_DIR=$(pwd)
VENV_DIR="$CURRENT_DIR/venv"

mkdir -p pcap_traces attack_logs analysis_results

# ====================================
# 1. Environment Setup
# ====================================
echo -e "${BLUE}[1/10]${NC} Setting up environment..."

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
pip install xgboost scikit-learn pandas numpy matplotlib seaborn scipy imbalanced-learn --quiet

echo -e "${GREEN}[✓]${NC} Environment ready"

# ====================================
# 2. Create Enhanced NS-3 Simulator
# ====================================
echo ""
echo -e "${BLUE}[2/10]${NC} Creating simulator with stealthy data exfiltration..."

cat > "$SCRATCH_DIR/smart-home-realistic.cc" << 'CPPEOF'
#!/bin/bash
# =============================================================================
#  CausalCIT 一键环境配置 + 实验运行脚本
#
#  用法:
#    bash setup.sh                          # 完整流程（环境 + 数据 + 实验）
#    bash setup.sh --env-only               # 仅创建环境、安装依赖、下载数据
#    bash setup.sh --run-only               # 仅在已有环境下运行实验
#    bash setup.sh --exp demo               # 仅运行 demo 实验
#    bash setup.sh --exp enhanced           # 仅运行增强实验 v2
#    bash setup.sh --exp ablation           # 仅运行消融实验
#    bash setup.sh --device cpu             # 强制使用 CPU
#    bash setup.sh --name my_env            # 自定义 conda 环境名
#
# =============================================================================
set -e

# ── 默认配置 ──────────────────────────────────────────────────
ENV_NAME="causalcit"
PYTHON_VER="3.10"
ENV_ONLY=false
RUN_ONLY=false
EXP_MODE="all"
DEVICE="auto"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 解析参数 ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-only)  ENV_ONLY=true; shift ;;
        --run-only)  RUN_ONLY=true; shift ;;
        --exp)       EXP_MODE="$2"; shift 2 ;;
        --device)    DEVICE="$2"; shift 2 ;;
        --name)      ENV_NAME="$2"; shift 2 ;;
        --python)    PYTHON_VER="$2"; shift 2 ;;
        -h|--help)
            echo "用法: bash setup.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --env-only    仅配置环境 + 下载数据，不运行实验"
            echo "  --run-only    仅运行实验（跳过环境配置）"
            echo "  --exp MODE    指定实验: demo | enhanced | ablation | all (默认 all)"
            echo "  --device DEV  cpu | cuda | auto (默认 auto)"
            echo "  --name NAME   conda 环境名 (默认 causalcit)"
            echo "  --python VER  Python 版本 (默认 3.10)"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 颜色输出 ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

header()  { echo -e "\n${CYAN}════════════════════════════════════════════════════════${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
warning() { echo -e "${YELLOW}⚠${NC} $1"; }
error()   { echo -e "${RED}✗${NC} $1"; }

# =============================================================================
#  Step 1: 环境配置
# =============================================================================
setup_env() {
    header "Step 1/4: 配置 Python 环境"

    # 检测 conda
    CONDA_EXE=""
    if command -v conda &>/dev/null; then
        CONDA_EXE="conda"
    elif [ -f "$HOME/miniconda3/bin/conda" ]; then
        CONDA_EXE="$HOME/miniconda3/bin/conda"
    elif [ -f "$HOME/anaconda3/bin/conda" ]; then
        CONDA_EXE="$HOME/anaconda3/bin/conda"
    fi

    if [ -n "$CONDA_EXE" ]; then
        success "检测到 conda: $CONDA_EXE"

        # 检查环境是否已存在
        if $CONDA_EXE env list | grep -q "^${ENV_NAME} "; then
            warning "conda 环境 '${ENV_NAME}' 已存在，跳过创建"
        else
            echo "  创建 conda 环境: ${ENV_NAME} (Python ${PYTHON_VER})"
            $CONDA_EXE create -n "$ENV_NAME" python="$PYTHON_VER" -y
        fi

        # 获取 conda 环境的 python 路径
        CONDA_ENV_DIR=$($CONDA_EXE env list | grep "^${ENV_NAME} " | awk '{print $NF}')
        PYTHON="$CONDA_ENV_DIR/bin/python"
        PIP="$CONDA_ENV_DIR/bin/pip"
    else
        warning "未检测到 conda，使用系统 Python"
        PYTHON="$(which python3 || which python)"
        PIP="$(which pip3 || which pip)"
    fi

    # 验证 Python
    echo "  Python: $($PYTHON --version 2>&1)"

    # 安装依赖
    header "Step 2/4: 安装 Python 依赖"
    echo "  安装位置: $SCRIPT_DIR/requirements.txt"
    $PIP install --upgrade pip -q
    $PIP install -r "$SCRIPT_DIR/requirements.txt"
    success "依赖安装完成"
}

# =============================================================================
#  Step 2: 下载数据集
# =============================================================================
download_data() {
    header "Step 3/4: 准备数据集"

    cd "$SCRIPT_DIR"
    $PYTHON download_data.py
}

# =============================================================================
#  Step 3: 运行实验
# =============================================================================
run_experiments() {
    header "Step 4/4: 运行实验"

    # 确定设备参数
    if [ "$DEVICE" = "auto" ]; then
        DEVICE_ARG=""
    else
        DEVICE_ARG="--device $DEVICE"
    fi

    cd "$SCRIPT_DIR"

    # ── Demo 实验 ──
    if [ "$EXP_MODE" = "all" ] || [ "$EXP_MODE" = "demo" ]; then
        echo ""
        echo "████████████████████████████████████████████████████████████████"
        echo "  实验: CausalCIT Demo (合成数据 + OOD)"
        echo "████████████████████████████████████████████████████████████████"

        cd "$SCRIPT_DIR/CausalCIT_demo"
        $PYTHON run_demo.py $DEVICE_ARG
        success "Demo 实验完成 → $(pwd)/output/"
    fi

    # ── 增强实验 v2 ──
    if [ "$EXP_MODE" = "all" ] || [ "$EXP_MODE" = "enhanced" ]; then
        echo ""
        echo "████████████████████████████████████████████████████████████████"
        echo "  实验: CausalCIT 增强实验 v2 (大模型 + 真实数据)"
        echo "████████████████████████████████████████████████████████████████"

        cd "$SCRIPT_DIR/CausalCIT_exp_v2"
        $PYTHON run_enhanced.py $DEVICE_ARG
        success "增强实验 v2 完成 → $(pwd)/output/"
    fi

    # ── 消融实验 ──
    if [ "$EXP_MODE" = "all" ] || [ "$EXP_MODE" = "ablation" ]; then
        echo ""
        echo "████████████████████████████████████████████████████████████████"
        echo "  实验: CausalCIT 消融实验"
        echo "████████████████████████████████████████████████████████████████"

        cd "$SCRIPT_DIR/CausalCIT_ablation"
        $PYTHON run_ablation.py $DEVICE_ARG
        success "消融实验完成 → $(pwd)/output/"
    fi
}

# ── 完成提示 ──────────────────────────────────────────────────
finish() {
    echo ""
    echo "══════════════════════════════════════════════════════════════════"
    echo "  全部完成！"
    echo "══════════════════════════════════════════════════════════════════"
    echo ""
    echo "  输出目录:"
    echo "    Demo:       $SCRIPT_DIR/CausalCIT_demo/output/"
    echo "    增强实验:   $SCRIPT_DIR/CausalCIT_exp_v2/output/"
    echo "    消融实验:   $SCRIPT_DIR/CausalCIT_ablation/output/"
    echo ""
}

# =============================================================================
#  主流程
# =============================================================================
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║          CausalCIT — 因果通道交互 Transformer                    ║"
    echo "║          一键环境配置 & 完整实验流程                              ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  项目根目录: $SCRIPT_DIR"
    echo "  运行模式:   EXP_MODE=$EXP_MODE  ENV_ONLY=$ENV_ONLY  RUN_ONLY=$RUN_ONLY"
    echo ""

    if [ "$RUN_ONLY" = false ]; then
        setup_env
        download_data
    fi

    if [ "$ENV_ONLY" = false ]; then
        run_experiments
        finish
    else
        echo ""
        success "环境配置完成！运行实验请执行: bash setup.sh --run-only"
    fi
}

main

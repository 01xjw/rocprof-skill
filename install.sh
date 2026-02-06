#!/bin/bash
# ============================================================================
# ROCprof Skill - One-Click Installation Script
# Supports Kimi Code CLI / Claude Code / Cursor / Codex / DeepSeek AI Agents
# ============================================================================
#
# Usage:
#   ./install.sh                # Default install to Kimi Code CLI
#   ./install.sh --cursor       # Install to Cursor
#   ./install.sh --deepseek     # Install to DeepSeek
#   ./install.sh --all          # Install to all supported Agents
#   ./install.sh --help         # Show help
#
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# Script directory (i.e. the rocprof-skill project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_NAME="rocprof-hip-profiling"

# Installation paths for each Agent
KIMI_PATH="$HOME/.config/agents/skills/$SKILL_NAME"
CLAUDE_PATH="$HOME/.claude/skills/$SKILL_NAME"
CURSOR_RULES_DIR="$HOME/.cursor/rules"
CURSOR_PATH="$CURSOR_RULES_DIR/$SKILL_NAME.md"
CODEX_PATH="$HOME/.codex/skills/$SKILL_NAME"
DEEPSEEK_PATH="$HOME/.deepseek/skills/$SKILL_NAME"

# ============================================================================
# Help
# ============================================================================

show_help() {
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║           ROCprof Skill Installation Script                  ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    ./install.sh [options]

Agent installation options:
    --kimi              Install to Kimi Code CLI (default)
    --claude            Install to Claude Code
    --cursor            Install to Cursor (via rules)
    --codex             Install to Codex (OpenAI)
    --deepseek          Install to DeepSeek
    --all               Install to all supported Agents

Other options:
    -t, --target PATH   Install to a specified directory
    -p, --project       Install to current project (.cursor/rules/)
    --uninstall         Uninstall (from all installed locations)
    --check             Check ROCm environment
    --status            View installation status
    -h, --help          Show this help message

Examples:
    ./install.sh                          # Default install to Kimi Code CLI
    ./install.sh --kimi --cursor          # Install to both Kimi + Cursor
    ./install.sh --deepseek               # Install to DeepSeek
    ./install.sh --all                    # Install to all Agents
    ./install.sh --project                # Install to current project
    ./install.sh -t ~/my-skills/rocprof   # Install to custom directory
    ./install.sh --status                 # View installation status
    ./install.sh --uninstall              # Uninstall

Supported Agents:
    Agent            Install Path
    ─────────────    ──────────────────────────────────────────
    Kimi Code CLI    ~/.config/agents/skills/rocprof-hip-profiling/
    Claude Code      ~/.claude/skills/rocprof-hip-profiling/
    Cursor           ~/.cursor/rules/rocprof-hip-profiling.md
    Codex (OpenAI)   ~/.codex/skills/rocprof-hip-profiling/
    DeepSeek         ~/.deepseek/skills/rocprof-hip-profiling/

EOF
}

# ============================================================================
# Environment Check
# ============================================================================

check_environment() {
    echo ""
    print_info "Checking ROCm environment..."
    echo ""

    local ok=true

    # Check ROCm
    if [ -d "/opt/rocm" ]; then
        local rocm_ver="unknown"
        if [ -f "/opt/rocm/.info/version" ]; then
            rocm_ver=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
        elif [ -f "/opt/rocm/include/rocm-core/rocm_version.h" ]; then
            rocm_ver=$(grep "ROCM_VERSION_STRING" /opt/rocm/include/rocm-core/rocm_version.h 2>/dev/null | awk -F'"' '{print $2}' || echo "unknown")
        fi
        print_success "ROCm: $rocm_ver (/opt/rocm)"
    else
        print_error "ROCm not installed (/opt/rocm does not exist)"
        ok=false
    fi

    # Check rocprof-sys
    if command -v rocprof-sys-run &> /dev/null; then
        print_success "rocprof-sys-run: $(which rocprof-sys-run)"
    elif [ -x "/opt/rocm/bin/rocprof-sys-run" ]; then
        print_warning "rocprof-sys-run: /opt/rocm/bin/rocprof-sys-run (not in PATH)"
        echo "         Add to PATH: export PATH=\$PATH:/opt/rocm/bin"
    else
        print_error "rocprof-sys-run not found"
        ok=false
    fi

    # Check GPU
    if command -v rocm-smi &> /dev/null; then
        local gpu_count=$(rocm-smi --showid 2>/dev/null | grep "GPU" | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            print_success "AMD GPU: $gpu_count device(s) detected"
        else
            print_warning "AMD GPU: no devices detected"
        fi
    elif [ -d "/sys/class/kfd/kfd/topology/nodes" ]; then
        local gpu_count=$(ls -d /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | wc -l)
        print_success "KFD nodes: $gpu_count"
    else
        print_warning "Cannot detect GPU (rocm-smi not available)"
    fi

    # Check Python
    if command -v python3 &> /dev/null; then
        local py_ver=$(python3 --version 2>&1)
        print_success "$py_ver"
    else
        print_warning "Python3 not installed (required for PyTorch analysis)"
    fi

    # Check PyTorch
    if python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        local hip_avail=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        if [ "$hip_avail" = "True" ]; then
            print_success "PyTorch HIP: available"
        else
            print_warning "PyTorch HIP: not available (torch.cuda.is_available() = False)"
        fi
    else
        print_warning "PyTorch not installed (optional)"
    fi

    # Check Triton
    if python3 -c "import triton; print(f'Triton {triton.__version__}')" 2>/dev/null; then
        print_success "Triton: installed"
    else
        print_warning "Triton not installed (optional, required for triton_runner)"
    fi

    echo ""
    if [ "$ok" = true ]; then
        print_success "Environment check passed"
    else
        print_error "Environment check failed, please install missing dependencies"
    fi
}

# ============================================================================
# Installation Core Logic
# ============================================================================

# Install to directory-based Agents (Kimi / Claude / Codex / DeepSeek)
install_to_dir() {
    local target_path="$1"
    local agent_name="$2"

    print_info "Installing to $agent_name: $target_path"

    mkdir -p "$target_path"

    # Core file: SKILL.md
    cp "$SCRIPT_DIR/SKILL.md" "$target_path/SKILL.md"

    # Scripts directory
    if [ -d "$SCRIPT_DIR/scripts" ]; then
        mkdir -p "$target_path/scripts"
        cp "$SCRIPT_DIR/scripts/"*.py "$target_path/scripts/" 2>/dev/null || true
        cp "$SCRIPT_DIR/scripts/"*.sh "$target_path/scripts/" 2>/dev/null || true
        chmod +x "$target_path/scripts/"*.sh 2>/dev/null || true
    fi

    # Examples directory
    if [ -d "$SCRIPT_DIR/examples" ]; then
        cp -r "$SCRIPT_DIR/examples" "$target_path/examples"
        chmod +x "$target_path/examples/"*.sh 2>/dev/null || true
    fi

    print_success "$agent_name installation complete: $target_path"
}

# Install to Cursor (rules file method)
install_to_cursor() {
    local target_path="$1"
    local rules_dir

    if [ "$target_path" = "__project__" ]; then
        # Project level: install to current directory's .cursor/rules/
        rules_dir="$(pwd)/.cursor/rules"
    else
        # User level: install to ~/.cursor/rules/
        rules_dir="$CURSOR_RULES_DIR"
    fi

    local rules_file="$rules_dir/$SKILL_NAME.md"

    print_info "Installing to Cursor rules: $rules_file"

    mkdir -p "$rules_dir"

    # Cursor rules directly uses SKILL.md content as the rules file
    cp "$SCRIPT_DIR/SKILL.md" "$rules_file"

    print_success "Cursor installation complete: $rules_file"
    echo "         Cursor will automatically load .md files from the rules directory on startup"
}

# ============================================================================
# Uninstall
# ============================================================================

uninstall_all() {
    print_info "Uninstalling ROCprof Skill..."
    echo ""

    local paths=("$KIMI_PATH" "$CLAUDE_PATH" "$CODEX_PATH" "$DEEPSEEK_PATH")
    local names=("Kimi Code CLI" "Claude Code" "Codex" "DeepSeek")

    for i in "${!paths[@]}"; do
        if [ -d "${paths[$i]}" ]; then
            rm -rf "${paths[$i]}"
            print_success "Uninstalled: ${names[$i]} (${paths[$i]})"
        fi
    done

    # Cursor rules file
    if [ -f "$CURSOR_PATH" ]; then
        rm -f "$CURSOR_PATH"
        print_success "Uninstalled: Cursor ($CURSOR_PATH)"
    fi

    # Project-level Cursor rules
    local project_rules="$(pwd)/.cursor/rules/$SKILL_NAME.md"
    if [ -f "$project_rules" ]; then
        rm -f "$project_rules"
        print_success "Uninstalled: Cursor project-level ($project_rules)"
    fi

    echo ""
    print_success "Uninstall complete"
}

# ============================================================================
# Status
# ============================================================================

show_status() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           ROCprof Skill Installation Status                 ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    # Kimi Code CLI
    if [ -f "$KIMI_PATH/SKILL.md" ]; then
        print_success "Kimi Code CLI:  Installed ($KIMI_PATH)"
    else
        echo -e "  ${RED}✗${NC} Kimi Code CLI:  Not installed"
    fi

    # Claude Code
    if [ -f "$CLAUDE_PATH/SKILL.md" ]; then
        print_success "Claude Code:    Installed ($CLAUDE_PATH)"
    else
        echo -e "  ${RED}✗${NC} Claude Code:    Not installed"
    fi

    # Cursor (user level)
    if [ -f "$CURSOR_PATH" ]; then
        print_success "Cursor (user):  Installed ($CURSOR_PATH)"
    else
        echo -e "  ${RED}✗${NC} Cursor (user):  Not installed"
    fi

    # Cursor (project level)
    local project_rules="$(pwd)/.cursor/rules/$SKILL_NAME.md"
    if [ -f "$project_rules" ]; then
        print_success "Cursor (project): Installed ($project_rules)"
    else
        echo -e "  ${RED}✗${NC} Cursor (project): Not installed"
    fi

    # Codex
    if [ -f "$CODEX_PATH/SKILL.md" ]; then
        print_success "Codex:          Installed ($CODEX_PATH)"
    else
        echo -e "  ${RED}✗${NC} Codex:          Not installed"
    fi

    # DeepSeek
    if [ -f "$DEEPSEEK_PATH/SKILL.md" ]; then
        print_success "DeepSeek:       Installed ($DEEPSEEK_PATH)"
    else
        echo -e "  ${RED}✗${NC} DeepSeek:       Not installed"
    fi

    echo ""
}

# ============================================================================
# Main
# ============================================================================

main() {
    local do_kimi=false
    local do_claude=false
    local do_cursor=false
    local do_codex=false
    local do_deepseek=false
    local do_all=false
    local do_project=false
    local do_check=false
    local do_uninstall=false
    local do_status=false
    local custom_path=""
    local any_agent=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            --kimi)
                do_kimi=true
                any_agent=true
                shift
                ;;
            --claude)
                do_claude=true
                any_agent=true
                shift
                ;;
            --cursor)
                do_cursor=true
                any_agent=true
                shift
                ;;
            --codex)
                do_codex=true
                any_agent=true
                shift
                ;;
            --deepseek)
                do_deepseek=true
                any_agent=true
                shift
                ;;
            --all)
                do_all=true
                any_agent=true
                shift
                ;;
            -p|--project)
                do_project=true
                any_agent=true
                shift
                ;;
            -t|--target)
                custom_path="$2"
                any_agent=true
                shift 2
                ;;
            --check)
                do_check=true
                shift
                ;;
            --uninstall)
                do_uninstall=true
                shift
                ;;
            --status)
                do_status=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Run ./install.sh --help for usage"
                exit 1
                ;;
        esac
    done

    # Special operations
    if [ "$do_check" = true ]; then
        check_environment
        exit 0
    fi

    if [ "$do_uninstall" = true ]; then
        uninstall_all
        exit 0
    fi

    if [ "$do_status" = true ]; then
        show_status
        exit 0
    fi

    # Default: install to Kimi Code CLI
    if [ "$any_agent" = false ]; then
        do_kimi=true
    fi

    # --all expands to all agents
    if [ "$do_all" = true ]; then
        do_kimi=true
        do_claude=true
        do_cursor=true
        do_codex=true
        do_deepseek=true
    fi

    # Check SKILL.md exists
    if [ ! -f "$SCRIPT_DIR/SKILL.md" ]; then
        print_error "SKILL.md not found, please ensure you run from the rocprof-skill project directory"
        exit 1
    fi

    # Banner
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║           ROCprof Skill Installer                            ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    local installed=0

    # Custom directory
    if [ -n "$custom_path" ]; then
        install_to_dir "$custom_path" "Custom directory"
        installed=$((installed + 1))
    fi

    # Kimi Code CLI
    if [ "$do_kimi" = true ]; then
        install_to_dir "$KIMI_PATH" "Kimi Code CLI"
        echo "         The skill will be auto-loaded when Kimi Code CLI starts"
        echo ""
        installed=$((installed + 1))
    fi

    # Claude Code
    if [ "$do_claude" = true ]; then
        install_to_dir "$CLAUDE_PATH" "Claude Code"
        echo ""
        installed=$((installed + 1))
    fi

    # Cursor
    if [ "$do_cursor" = true ]; then
        install_to_cursor "user"
        echo ""
        installed=$((installed + 1))
    fi

    # Cursor project level
    if [ "$do_project" = true ]; then
        install_to_cursor "__project__"
        echo ""
        installed=$((installed + 1))
    fi

    # Codex
    if [ "$do_codex" = true ]; then
        install_to_dir "$CODEX_PATH" "Codex"
        echo ""
        installed=$((installed + 1))
    fi

    # DeepSeek
    if [ "$do_deepseek" = true ]; then
        install_to_dir "$DEEPSEEK_PATH" "DeepSeek"
        echo ""
        installed=$((installed + 1))
    fi

    # Done
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
    print_success "Installation complete! Installed to $installed target(s)"
    echo ""
    echo "View installation status: ./install.sh --status"
    echo "Check ROCm environment:   ./install.sh --check"
    echo "Uninstall:                ./install.sh --uninstall"
    echo ""
}

main "$@"

#!/usr/bin/env bash
# ==============================================================================
# Script: init_github_repo.sh
# Description: Automates local git initialization, initial commit, GitHub remote
#              repository creation via GitHub CLI (`gh`), and branch push.
# ==============================================================================

set -euo pipefail

# Color palette
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Defaults
REPO_NAME="os-autofix-engine"
VISIBILITY="--public"
REMOTE_NAME="origin"
DEFAULT_BRANCH="main"
ORG=""

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    -n, --name NAME         Repository name (default: os-autofix-engine)
    -p, --private           Create private repository (default: public)
    --public                Create public repository
    -o, --org ORG           GitHub organization to create under
    -r, --remote REMOTE     Remote name (default: origin)
    -b, --branch BRANCH     Default branch name (default: main)
    -h, --help              Show this help message

Example:
    ./scripts/init_github_repo.sh --name os-autofix-engine --public
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)
            REPO_NAME="$2"
            shift 2
            ;;
        -p|--private)
            VISIBILITY="--private"
            shift
            ;;
        --public)
            VISIBILITY="--public"
            shift
            ;;
        -o|--org)
            ORG="$2"
            shift 2
            ;;
        -r|--remote)
            REMOTE_NAME="$2"
            shift 2
            ;;
        -b|--branch)
            DEFAULT_BRANCH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Step 1: Verify prerequisites
print_info "Checking prerequisites..."
if ! command -v git &> /dev/null; then
    print_error "git is not installed or not in PATH."
    exit 1
fi

if ! command -v gh &> /dev/null; then
    print_error "GitHub CLI (gh) is not installed or not in PATH."
    exit 1
fi

# Step 2: Initialize local git repository
if [ ! -d ".git" ]; then
    print_info "Initializing git repository..."
    git init -b "$DEFAULT_BRANCH"
else
    print_info "Git repository already initialized."
    git checkout -B "$DEFAULT_BRANCH" 2>/dev/null || true
fi

# Step 3: Stage and commit files
print_info "Staging repository files..."
git add .

if git diff --cached --quiet; then
    print_info "No staged changes detected. Repository working tree is clean."
else
    print_info "Creating initial commit..."
    git commit -m "feat: initial commit for os-autofix-engine autonomous harness"
fi

# Step 4: Check GitHub CLI authentication
print_info "Checking GitHub authentication status..."
if ! gh auth status &>/dev/null; then
    print_warn "GitHub CLI is not logged in. You can run 'gh auth login' to authenticate."
fi

# Step 5: Create remote repository via gh
TARGET_REPO="$REPO_NAME"
if [ -n "$ORG" ]; then
    TARGET_REPO="$ORG/$REPO_NAME"
fi

print_info "Checking if remote repository '$TARGET_REPO' exists on GitHub..."
if gh repo view "$TARGET_REPO" &>/dev/null; then
    print_info "Repository '$TARGET_REPO' already exists on GitHub."
    REMOTE_URL=$(gh repo view "$TARGET_REPO" --json url -q .url).git
    if git remote get-url "$REMOTE_NAME" &>/dev/null; then
        git remote set-url "$REMOTE_NAME" "$REMOTE_URL"
    else
        git remote add "$REMOTE_NAME" "$REMOTE_URL"
    fi
else
    print_info "Creating GitHub repository '$TARGET_REPO' ($VISIBILITY)..."
    gh repo create "$TARGET_REPO" "$VISIBILITY" --source=. --remote="$REMOTE_NAME" --push || {
        print_warn "gh repo create command returned non-zero code. Attempting manual push..."
    }
fi

# Step 6: Push current branch
print_info "Pushing branch '$DEFAULT_BRANCH' to remote '$REMOTE_NAME'..."
if git push -u "$REMOTE_NAME" "$DEFAULT_BRANCH" 2>/dev/null; then
    print_success "Repository successfully pushed to GitHub: $TARGET_REPO"
else
    print_warn "Automatic push skipped or failed. Run 'git push -u $REMOTE_NAME $DEFAULT_BRANCH' manually."
fi

print_success "GitHub setup completed."

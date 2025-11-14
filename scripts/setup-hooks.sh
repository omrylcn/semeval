#!/bin/bash
# Setup Git Hooks
# Installs pre-push hook that runs tests before pushing to main

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

echo "🔧 Setting up Git hooks..."
echo ""

# Check if we're in a git repository
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo "❌ Error: Not a git repository"
    exit 1
fi

# Install pre-push hook
echo "📦 Installing pre-push hook..."
cp "$SCRIPT_DIR/hooks/pre-push" "$HOOKS_DIR/pre-push"
chmod +x "$HOOKS_DIR/pre-push"
echo "✅ pre-push hook installed"

echo ""
echo "✅ Git hooks setup complete!"
echo ""
echo "The pre-push hook will automatically run 'make pre-merge'"
echo "before pushing to main branch."
echo ""
echo "To bypass (NOT RECOMMENDED):"
echo "  git push --no-verify"
echo ""

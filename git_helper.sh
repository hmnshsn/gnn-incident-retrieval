#!/bin/bash
# Git Helper Script - Automatically uses credentials from .env file
# Usage: ./git_helper.sh push
#        ./git_helper.sh pull

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found!"
    echo "Please create .env file with your GitHub credentials."
    echo "See .env.example for template."
    exit 1
fi

# Check if required variables are set
if [ -z "$GITHUB_USERNAME" ] || [ -z "$GITHUB_PAT" ] || [ -z "$GITHUB_REPO" ]; then
    echo "Error: Missing required environment variables in .env"
    echo "Required: GITHUB_USERNAME, GITHUB_PAT, GITHUB_REPO"
    exit 1
fi

# Construct authenticated URL
GITHUB_URL="https://${GITHUB_USERNAME}:${GITHUB_PAT}@github.com/${GITHUB_REPO}.git"

# Execute git command
case "$1" in
    push)
        echo "Pushing to GitHub..."
        git push "$GITHUB_URL" $(git branch --show-current) --force
        ;;
    pull)
        echo "Pulling from GitHub..."
        git pull "$GITHUB_URL" $(git branch --show-current)
        ;;
    clone)
        echo "Cloning from GitHub..."
        git clone "$GITHUB_URL"
        ;;
    remote)
        echo "Setting remote origin..."
        git remote remove origin 2>/dev/null
        git remote add origin "$GITHUB_URL"
        echo "Remote origin set successfully"
        ;;
    *)
        echo "Usage: $0 {push|pull|clone|remote}"
        echo ""
        echo "Commands:"
        echo "  push   - Push current branch to GitHub"
        echo "  pull   - Pull current branch from GitHub"
        echo "  clone  - Clone repository from GitHub"
        echo "  remote - Set remote origin with credentials"
        exit 1
        ;;
esac

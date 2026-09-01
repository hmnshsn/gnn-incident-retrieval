#!/bin/bash
# Git Setup Script - Configure local git user settings
# Usage: bash git_setup.sh

echo "Setting up local git configuration..."

# Set local git user name and email
git config user.name "himanshu.pal"
git config user.email "himanshu.pal@servicenow.com"

echo "Git configuration complete!"
echo "User: $(git config user.name)"
echo "Email: $(git config user.email)"
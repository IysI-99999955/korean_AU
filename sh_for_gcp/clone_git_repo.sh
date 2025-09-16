#!/bin/bash

# git, curl, vim 설치
sudo apt-get update && sudo apt-get install -y curl git vim && echo "curl, git, vim installed."

# GitHub 리포지토리 클론
# read -p "Enter the GitHub repository URL to clone: " GIT_URL
GIT_URL = "https://github.com/IysI-99999955/korean_AU.git"
git clone $GIT_URL && echo "Repository cloned from $GIT_URL."
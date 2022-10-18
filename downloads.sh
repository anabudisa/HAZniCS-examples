#!/bin/bash
mkdir downloads
cd downloads || exit

# google tool that also work with google drive
pip3 install gdown
# gdown is installed under .local/bin
export PATH=$PATH:$HOME/.local/bin

# download meshes for neuron and brain
gdown --folder https://drive.google.com/drive/folders/1vjk8JQwQEiIyuXSZqH9jq2oAxuK95s8C


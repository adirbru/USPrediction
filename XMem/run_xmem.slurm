#!/bin/bash
#SBATCH --job-name=xmem
#SBATCH --output=logs/xmem_%j.out
#SBATCH --error=logs/xmem_%j.err
#SBATCH --partition=all          
#SBATCH --gres=gpu:1             
#SBATCH --cpus-per-task=4              
#SBATCH --mem=16G                      
#SBATCH --time=01:00:00                

python3 -m pip install -r requirements.txt
python3 eval.py --model XMem.pth --output "/home/adir.bruchim/output" --dataset G --generic_path "/home/adir.bruchim/USPrediction/code/dataset/processed/custom_data_root" --mem_every 5
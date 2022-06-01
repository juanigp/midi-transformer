#!/bin/bash -l
#SBATCH --mem=200G
#SBATCH --cpus-per-task=64
#SBATCH --time=5-12:00:00
#SBATCH --output=./sbatch-out/train_%j.txt
#SBATCH --job-name=train-midi-transformer
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ada00

conda activate midi-env
python3 clm_midi_dataset.py -cp configs -cn lm_config.yml
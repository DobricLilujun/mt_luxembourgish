#!/bin/bash
#SBATCH --job-name=translation_job       # Job name
#SBATCH --output=translation_output_%j.log  # Output file (logs), %j is the job ID
#SBATCH --error=translation_error_%j.log   # Error file (logs)
#SBATCH --time=015:00:00                  # Maximum wall time (2 hours)
#SBATCH --partition=gpu                   # GPU partition (adjust as needed)
#SBATCH --gres=gpu:2                      # Request 1 GPU
#SBATCH --mem=16G                         # Memory allocation (16 GB)
#SBATCH --cpus-per-task=4                # Number of CPUs to allocate per task


# Activate the appropriate virtual environment (if needed)
conda init
conda activate /home/users/luli/.conda/envs/mt_lux_env

# Run the Python script with the appropriate command-line arguments
python /home/users/luli/project/mt_luxembourgish/nllb_running_inference.py \
  --model_name /mnt/lscratch/users/luli/model/nllb-200-3.3B/ \
  --src_lang ltz_Latn \
  --tgt_lang eng_Latn \
  --device cuda \
  --max_length 480 \
  --batch_size 300 \
  --text_column subsentence \
  --prefix translation_nllb_ \
  --input_file /home/users/luli/project/mt_luxembourgish/NC_lux_subsentences.csv

# End of the script

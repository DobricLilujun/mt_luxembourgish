#!/bin/bash
#SBATCH --job-name=translation_job       # Job name
#SBATCH --output=translation_output_%j.log  # Output file (logs), %j is the job ID
#SBATCH --error=translation_error_%j.log   # Error file (logs)
#SBATCH --time=015:00:00                  # Maximum wall time (2 hours)
#SBATCH --partition=gpu                   # GPU partition (adjust as needed)
#SBATCH --gres=gpu:4                    # Request GPU
#SBATCH --mem=48G                         # Memory allocation (16 GB)
#SBATCH --cpus-per-task=4                # Number of CPUs to allocate per task


# Activate the appropriate virtual environment (if needed)
conda init
conda activate /home/users/luli/.conda/envs/mt_lux_env

# /home/users/luli/.conda/envs/mt_lux_env/bin/python /home/users/luli/project/mt_luxembourgish/localLLM_running_inference.py --model_name /mnt/lscratch/users/luli/model/Meta-Llama-3-8B-Instruct/ --if_loading_quantization True --current_load_in_4bit True --batch_size 2 --text_column subsentence --prefix translation_LLM_huggingface_pipeline_ --input_file /home/users/luli/project/mt_luxembourgish/NC_lux_subsentences_test.csv --is_new_file False --device auto

# Run the Python script with the appropriate command-line arguments
/home/users/luli/.conda/envs/mt_lux_env/bin/python /home/users/luli/project/mt_luxembourgish/localLLM_running_inference.py \
  --model_name /mnt/lscratch/users/luli/model/gemma-2-27b-it/ \
  --batch_size 10 \
  --text_column subsentence \
  --prefix translation_LLM_huggingface_pipeline_ \
  --input_file /home/users/luli/project/mt_luxembourgish/NC_lux_subsentences.csv \
  --is_new_file True

# End of the script

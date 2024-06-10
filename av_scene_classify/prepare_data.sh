#!/bin/bash

#SBATCH --job-name=train
#SBATCH -p dgx2
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --gres gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

__conda_setup="$('/lustre/home/acct-stu/stu282/Tools/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/lustre/home/acct-stu/stu282/Tools/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/lustre/home/acct-stu/stu282/Tools/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/lustre/home/acct-stu/stu282/Tools/miniconda3/bin:$PATH"
    fi
fi

export PATH="/lustre/home/acct-stu/stu282/Tools/miniconda3/envs/3.7:$PATH"
export LD_LIBRARY_PATH=/lustre/home/acct-stu/stu282/Tools/miniconda3/envs/py3.7/lib/:$LD_LIBRARY_PATH
conda config --set auto_activate_base false
conda init
conda activate py3.7

if [ ! $# -eq 1 ]; then
    echo -e "Usage: $0 <raw_data_dir>"
    exit 1
fi

DATA=$1

if [ ! -d data ]; then
    mkdir data
fi

cd data

if [ ! -d feature ]; then
    mkdir feature
fi

# prepare data
python split_data.py --input_path "$DATA/evaluation_setup/fold1_train.csv" \
    --output_path "./evaluation_setup"

python extract_openl3_emb.py --input_path './evaluation_setup/train.csv' \
    --dataset_path $DATA \
    --output_path './feature' \
    --split 'train'

python extract_openl3_emb.py --input_path './evaluation_setup/val.csv' \
    --dataset_path $DATA \
    --output_path './feature' \
    --split 'val'

python extract_openl3_emb.py --input_path "$DATA/evaluation_setup/fold1_evaluate.csv" \
    --dataset_path $DATA \
    --output_path './feature' \
    --split 'test'

python compute_mean_std.py --input_path './feature'

cd ..

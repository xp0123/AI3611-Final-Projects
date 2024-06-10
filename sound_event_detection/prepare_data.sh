#!/bin/bash

#SBATCH --job-name=feature
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -n 1
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

export PATH="/lustre/home/acct-stu/stu282/Tools/miniconda3/envs/py3.10.11:$PATH"
conda config --set auto_activate_base false
conda init
conda activate py3.10.11

if [ ! $# -eq 1 ]; then
    echo -e "Usage: $0 <raw_data_dir>"
    exit 1
fi


cd data


#module load miniconda3
#__conda_setup="$('/dssg/opt/icelake/linux-centos8-icelake/gcc-11.2.0/miniconda3-4.10.3-f5dsmdmzng2ck6a4otduqwosi22kacfl/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#eval "$__conda_setup"
#conda activate pytorch

DATA=$1
mkdir {dev,eval,metadata}

# development set
echo "Preparing development set"
python prepare_wav_csv.py "$DATA/audio/train/weak" "dev/wav.csv"
python extract_feature.py "dev/wav.csv" "dev/feature.h5" --sr 44100 --num_worker 1
ln -s $(realpath "$DATA/label/train/weak.csv") "dev/label.csv"

# evaluation set
echo "Preparing evaluation set"
python prepare_wav_csv.py "$DATA/audio/eval" "eval/wav.csv"
python extract_feature.py "eval/wav.csv" "eval/feature.h5" --sr 44100 --num_worker 1
ln -s $(realpath "$DATA/label/eval/eval.csv") "eval/label.csv"

cp "$DATA/label/class_label_indices.txt" "metadata/class_label_indices.txt"

cd ..

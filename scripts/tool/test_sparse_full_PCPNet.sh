#!/bin/sh

export PYTHONPATH=./
PYTHON=python3

TEST_CODE=test_sparse_full_PCPNet.py

dataset=pcpnet
exp_name=$1
config_name=$2
checkpoint=$3
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=results/${exp_name}
config=${exp_dir}/${config_name}.yaml

mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best

now=$(date +"%Y%m%d_%H%M%S")
cp tool/test_sparse_full_PCPNet.sh  tool/${TEST_CODE} ${exp_dir}

#: '
$PYTHON -u ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  save_folder ${result_dir}/${checkpoint} \
  model_path ${model_dir}/model_${checkpoint}.pth \
  2>&1 | tee ${exp_dir}/test_${checkpoint}-$now.log
#'


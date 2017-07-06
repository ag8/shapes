echo "Launching script $2 on GPU $1"
CUDA_VISIBLE_DEVICES=$1 nohup python $2 & unbuffer tail -f nohup.out &
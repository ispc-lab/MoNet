python train.py --batch_size 4 --epochs 100 --lr 0.001 --seed 1 --gpu GPU \
--root DATA_ROOT --npoints 16384 --rnn RNN --input_num 5 --pred_num 5 --dataset DATASET \
--ckpt_dir ./ckpt --multi_gpu --runname RUNNAME --wandb_dir ../wandb --use_wandb
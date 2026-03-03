TRAIN_FLAGS="--batch_size 15 --lr 1e-4 --save_interval 10000 --weight_decay 0.05 --dropout 0.0"
MODEL_FLAGS="--image_size 128 --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 500 --learn_sigma True --noise_schedule linear --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
GPU=0

echo "Training diffusor"
python celeba-train-diffusion.py $TRAIN_FLAGS $MODEL_FLAGS --gpus $GPU\
                                              --output_path 'ddpm_cas2_1'
                                              
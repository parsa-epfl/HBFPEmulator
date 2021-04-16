# wide-resnet for svhn
python main.py --type cnn --arch wideresnet16 --data svhn \
       --wideresnet_widen_factor 8\
       --lr 0.01 --lr_decay True \
       --lr_decay_epochs 24,48,64 --num_epochs 80 \
       --use_nesterov True --momentum 0.9 --weight_decay 5e-4\
       --batch_size 128 \
       --avg_model True  --lr_warmup True \
       --num_workers 2 --eval_freq 1  --reshuffle_per_epoch True \
       --lr_lars False --lr_scale True \
       --world_size 2 --device gpu --save_all_models True \
       --num_format bfp --rounding_mode stoc --mant_bits 7 --bfp_tile_size 24  --weight_mant_bits 15

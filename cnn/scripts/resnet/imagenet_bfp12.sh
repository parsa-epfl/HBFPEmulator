python main.py --type cnn --arch resnet50 --avg_model True \
    --data imagenet --use_lmdb_data True --data_dir /mlodata1/tlin/dataset/ILSVRC/ \
    --num_workers 2 --eval_freq 1 --num_epochs 90 --reshuffle_per_epoch True \
    --lr 0.1 --lr_scale True --lr_warmup True --lr_decay True --lr_decay_epochs 30,60,80 --lr_lars False \
    --use_nesterov True --momentum 0.9 --weight_decay 1e-4 --batch_size 64 \
    --world_size 2 --device gpu --save_all_models True \
    --num_format bfp --rounding_mode stoc --mant_bits 11 --bfp_tile_size 24  --weight_mant_bits 15

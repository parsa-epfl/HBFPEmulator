CUDA_VISIBLE_DEVICES=0 python main.py --type lstm --device gpu --batch_size 20\
                    --data lstm/data/penn --dropouti 0.4 --dropouth 0.25\
                    --seed 141 --epoch 500 --save bfp8.pt \
                    --num_format bfp --mant_bits 8 --bfp_tile_size 24 --weight_mant_bits 16

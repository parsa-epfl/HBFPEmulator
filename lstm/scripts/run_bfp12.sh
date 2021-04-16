CUDA_VISIBLE_DEVICES=1 python main.py --type lstm --device gpu  --batch_size 20\
                    --data lstm/data/wikitext-2 --dropouti 0.4 --dropouth 0.25\
                    --seed 141 --epoch 500 --save bfp12.pt \
                    --num_format bfp --mant_bits 12 --bfp_tile_size 24 --weight_mant_bits 16

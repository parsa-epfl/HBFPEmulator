CUDA_VISIBLE_DEVICES=0 python main.py --type lstm --device gpu --batch_size 20\
                    --data lstm/data/penn --dropouti 0.4 --dropouth 0.25\
                    --seed 141 --epoch 500 --save fp32.pt

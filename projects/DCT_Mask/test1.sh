CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train_net.py --config-file configs/DCT/R50_1x_dct_300_l1_0_007_4conv.yaml --eval-only \
                    --num-gpus 8 --resume \
                    --dist-url "tcp://127.0.0.1:6017"

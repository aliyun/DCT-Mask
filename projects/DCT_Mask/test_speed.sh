CUDA_VISIBLE_DEVICES=0 \
python train_net.py --config-file configs/DCT/R50_1x_dct_300_l1_0_007_4conv.yaml --eval-only \
                    --num-gpus 1 --resume \
                    --dist-url "tcp://127.0.0.1:6017" \
                    SOLVER.IMS_PER_BATCH 1


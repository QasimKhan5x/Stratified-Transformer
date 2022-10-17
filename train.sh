MASTER_ADDR=localhost
MASTER_PORT=12355
NUM_TRAINERS=4

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS \
    train.py --config config/s3dis/s3dis_swin3d_transformer.yaml

# python3 train2.py --config config/s3dis/s3dis_stratified_transformer.yaml
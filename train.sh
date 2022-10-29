MASTER_ADDR=localhost
MASTER_PORT=12355
NUM_TRAINERS=5

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS \
    train.py --config config/s3dis/s3dis_gc_transformer.yaml
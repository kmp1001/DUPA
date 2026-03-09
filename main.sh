#!/bin/bash

export NCCL_HOSTID=${MY_POD_NAME}
export MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
export MASTER_PORT=${ARNOLD_WORKER_0_PORT}
export NODE_RANK=${ARNOLD_ID}
export NUM_NODES=${ARNOLD_WORKER_NUM}

python3 main.py fit -c $1 --trainer.num_nodes $NUM_NODES
# for pid in $(ps -ef | grep "yaml" | grep -v "grep" | awk '{print $2}'); do kill -9 $pid; done
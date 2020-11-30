FALSE=0
TRUE=1

# assign global devices
export CUDA_VISIBLE_DEVICES='0'

# select from: ['cifar10', 'cifar100']
DATASET='cifar100'
DATA_PATH='/home/zhaoyu/Datasets/cifar100'

# network model type and index
NET='resnet'
NET_INDEX=20

# training parameters
NUM_EPOCH=600
BATCH_SIZE=256
STD_BATCH_SIZE=256
STD_INIT_LR=1e-1

# distillation switch
DST_FLAG=${TRUE}

# prune switch
PRUNE_FLAG=${TRUE}

NET_DATASET=${NET}${NET_INDEX}_${DATASET}

BASIC_ARGUMENTS="--dataset ${DATASET}
                 --data_path ${DATA_PATH}
                 --net ${NET}
                 --net_index $((NET_INDEX))
                 --num_epoch $((NUM_EPOCH))
                 --batch_size ${BATCH_SIZE}
                 --std_batch_size ${STD_BATCH_SIZE}
                 --std_init_lr ${STD_INIT_LR}"

WORKROOT='workdir'
# append distillation arguments
DST_ARGUMENTS=" --dst_flag ${DST_FLAG} "
if [ ${DST_FLAG} == ${TRUE} ]; then
	TEACHER_NET='resnet'
	TEACHER_NET_INDEX=$((NET_INDEX))
	DST_TEMPERATURE=2
	DST_LOSS_WEIGHT=4
	TEACHER_DIR=${WORKROOT}/${NET_DATASET}/teacher
	mkdir -p ${TEACHER_DIR}
	DST_ARGUMENTS+="--teacher_net ${TEACHER_NET}
                    --teacher_net_index ${TEACHER_NET_INDEX}
                    --dst_temperature ${DST_TEMPERATURE}
                    --dst_loss_weight ${DST_LOSS_WEIGHT}
                    --teacher_dir ${TEACHER_DIR}"
fi

# append working directories arguments
FULL_DIR=${WORKROOT}/${NET_DATASET}/full
LOG_DIR=${WORKROOT}/${NET_DATASET}/log
mkdir -p ${FULL_DIR} ${LOG_DIR}
DIR_ARGUMENTS=" --full_dir ${FULL_DIR} --log_dir ${LOG_DIR} "
if [ ${PRUNE_FLAG} == ${TRUE} ]; then
	SLIM_DIR=${WORKROOT}/${NET_DATASET}/slim
	SLIM_DIR=${WORKROOT}/${NET_DATASET}/slim
	WARMUP_DIR=${WORKROOT}/${NET_DATASET}/warmup
	SEARCH_DIR=${WORKROOT}/${NET_DATASET}/search
	mkdir -p ${SLIM_DIR} ${WARMUP_DIR} ${SEARCH_DIR}
	DIR_ARGUMENTS+="--prune_flag ${PRUNE_FLAG}
	                --slim_dir ${SLIM_DIR}
	                --warmup_dir ${WARMUP_DIR}
	                --search_dir ${SEARCH_DIR}"
fi

BASIC_ARGUMENTS+=${DST_ARGUMENTS}
BASIC_ARGUMENTS+=${DIR_ARGUMENTS}
echo python -u main.py ${BASIC_ARGUMENTS}
TIME_TAG=`date +"%Y%m%d_%H%M"`
LOG_FILE=${LOG_DIR}/${TIME_TAG}.txt
python -u main.py ${BASIC_ARGUMENTS} 2>&1 | tee ${LOG_FILE}

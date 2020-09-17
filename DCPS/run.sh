FALSE=0
TRUE=1

# assign global devices
export CUDA_VISIBLE_DEVICES='0,1'

# select from: ['cifar10', 'cifar100']
DATASET='cifar100' 

# network model type and index
NET='resnet'
NET_INDEX=20
NET_NAME=${NET}${NET_INDEX}

# training parameters
BATCH_SIZE=128
STD_BATCH_SIZE=128
STD_INIT_LR=1e-1

# distillation switch
DST_FLAG=${TRUE}

# prune switch
PRUNE_FLAG=${TRUE}

BASIC_ARGUMENTS="--dataset ${DATASET}
                 --net ${NET}
				 --net_index ${NET_INDEX}
                 --batch_size ${BATCH_SIZE}
				 --std_batch_size ${BATCH_SIZE}
				 --std_init_lr ${STD_INIT_LR}"

# append distillation arguments
DST_ARGUMENTS="--dst_flag ${DST_FLAG}"
if [ ${DST_FLAG} == ${TRUE} ]; then
	TEACHER_NET='resnet'
	TEACHER_NET_INDEX=20
	DST_TEMPERATURE=4
	DST_LOSS_WEIGHT=4
	DST_ARGUMENTS+=" --teacher_net ${TEACHER_NET}
                     --teacher_net_index ${TEACHER_NET_INDEX}
                     --dst_temperature ${DST_TEMPERATURE}
                     --dst_loss_weight ${DST_LOSS_WEIGHT}"
fi

# append working directories arguments
WORKROOT='workdir'
FULL_DIR=${WORKROOT}/${NET_NAME}/full
LOG_DIR=${WORKROOT}/${NET_NAME}/log
mkdir -p ${FULL_DIR} ${LOG_DIR}
DIR_ARGUMENTS="--full_dir ${FULL_DIR} --log_dir ${LOG_DIR}"
if [ ${PRUNE_FLAG} == ${TRUE} ]; then
	SLIM_DIR=${WORKROOT}/${NET_NAME}/slim
	WARMUP_DIR=${WORKROOT}/${NET_NAME}/warmup
	SEARCH_DIR=${WORKROOT}/${NET_NAME}/search
	TEACHER_DIR=${WORKROOT}/${NET_NAME}/teacher
	mkdir -p ${SLIM_DIR} ${WARMUP_DIR} ${SEARCH_DIR} ${TEACHER_DIR}
	DIR_ARGUMENTS+=" --slim_dir ${SLIM_DIR}
	                 --warmup_dir ${WARMUP_DIR}
	                 --search_dir ${SEARCH_DIR}
	                 --teacher_dir ${TEACHER_DIR}"
fi

echo ${BASIC_ARGUMENTS} ${DST_ARGUMENTS} ${DIR_ARGUMENTS}
#python -u main.py ${BASIC_ARGUMENTS} ${DST_ARGUMENT} ${DIR_ARGUMENT}

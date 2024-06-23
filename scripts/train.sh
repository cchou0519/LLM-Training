PARTITION=
ACCOUNT=
GPUS_PER_NODE=
CPUS_PER_TASK=
EXTRA_ARGS=(
    --no-requeue
)
CONFIG=
CKPT_PATH=null

# export TRITON_PTXAS_PATH=$CONDA_PREFIX/bin/ptxas
# export TRITON_CUOBJDUMP_PATH=$CONDA_PREFIX/bin/cuobjdump
# export TRITON_NVDISASM_PATH=$CONDA_PREFIX/bin/nvdisasm

COMMAND=(
    srun llm-training fit
    --config $CONFIG
    --trainer.num_nodes $NODES
    --ckpt_path $CKPT_PATH
)

COMMAND=${COMMAND[@]}

echo $COMMAND

SBATCH_ARGS=(
    --job-name $JOB_NAME
    --partition $PARTITION
    --gpus-per-node $GPUS_PER_NODE
    --cpus-per-task $CPUS_PER_TASK
    --ntasks-per-node $GPUS_PER_NODE
    --account $ACCOUNT
    --nodes $NODES
)
SBATCH_ARGS+=(${EXTRA_ARGS[@]})
SBATCH_ARGS=${SBATCH_ARGS[@]}

SBATCH_OUTPUT=$(sbatch $SBATCH_ARGS --wrap "$COMMAND")

echo $SBATCH_OUTPUT

if [[ $SBATCH_OUTPUT != "Submitted batch job"* ]];
then
    exit
fi

JOB_ID=$(echo $SBATCH_OUTPUT | sed "s/Submitted batch job //")

SATTACH_START_TIME=$SECONDS
while [ $(($SECONDS - $SATTACH_START_TIME)) -le 30 ]
do
    SATTACH_OUTPUT=$(sattach $JOB_ID.0 2>&1 | tee /dev/tty)

    if [[ $SATTACH_OUTPUT == *"Job/step already completing or completed"* ]] \
    || [[ $SATTACH_OUTPUT == *"Socket timed out on send/recv operation"* ]] \
    || [[ $SATTACH_OUTPUT == *"does not look like a jobid"* ]];
    then
        break
    fi
    sleep 1
done

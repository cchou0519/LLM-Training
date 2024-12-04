# export TRITON_PTXAS_PATH=$CONDA_PREFIX/bin/ptxas
# export TRITON_CUOBJDUMP_PATH=$CONDA_PREFIX/bin/cuobjdump
# export TRITON_NVDISASM_PATH=$CONDA_PREFIX/bin/nvdisasm

JOB_NAME=
PARTITION=
ACCOUNT=
NODES=
GPUS_PER_NODE=
CPUS_PER_TASK=
EXTRA_ARGS=(
)
CONFIG=null
CKPT_PATH=null


COMMAND=(
    srun llm-training fit
    --config $CONFIG
    --trainer.num_nodes $NODES
    --ckpt_path $CKPT_PATH
)

COMMAND=${COMMAND[@]}

echo $COMMAND

SBATCH_ARGS=(
    --partition $PARTITION
    --gpus-per-node $GPUS_PER_NODE
    --cpus-per-task $CPUS_PER_TASK
    --ntasks-per-node $GPUS_PER_NODE
    --account $ACCOUNT
    --nodes $NODES
)

if [[ $JOB_NAME ]];
then
    SBATCH_ARGS+=(--job-name $JOB_NAME)
fi

SBATCH_ARGS+=(${EXTRA_ARGS[@]})
SBATCH_ARGS=${SBATCH_ARGS[@]}

SBATCH_OUTPUT=$(sbatch $SBATCH_ARGS --wrap "$COMMAND")

echo $SBATCH_OUTPUT

if [[ $SBATCH_OUTPUT != "Submitted batch job"* ]];
then
    exit
fi

JOB_ID=$(echo $SBATCH_OUTPUT | sed "s/Submitted batch job //")

echo "Waiting for the job to start"
while [[ $JOB_STATE != "RUNNING" ]]
do
    JOB_STATE=$(squeue -j $JOB_ID -h -o %T)
    sleep 1
done

echo "The job is running, trying to attach to the output stream ..."
sleep 3


while [[ $JOB_STATE == "RUNNING" ]]
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

# while [[ $JOB_STATE == "RUNNING" ]]
# do
#     JOB_STATE=$(squeue -j $JOB_ID -h -o %T)
#     tail -f slurm-$JOB_ID.out
#     sleep 1
# done

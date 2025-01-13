#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --account=PROJECT_NUMBER
#SBATCH --mail-type=ALL

if [[ -z "$SLURM_JOB_ID" ]]; then
  PARTITION="gpusmall"
  TIME="2:00:00"
  NUM_GPUS=1
  MEM=8
  GRES_GPU="gpu:a100:$NUM_GPUS"
  DYNAMIC_JOBNAME="$1"
  shift  
  JOB_SUBMISSION_OUTPUT=$(sbatch --job-name="$DYNAMIC_JOBNAME" --time="$TIME" --gres="$GRES_GPU" --mem="$MEM"G --partition="$PARTITION" -o "logs/${DYNAMIC_JOBNAME}-%j.log" "$0" "$@")
  echo "Submission output: $JOB_SUBMISSION_OUTPUT"
  JOB_ID=$(echo "$JOB_SUBMISSION_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
  LOG_FILE="logs/${DYNAMIC_JOBNAME}-${JOB_ID}.log"
  touch $LOG_FILE
  echo "tail -f $LOG_FILE"
  tail -f "$LOG_FILE"
  exit $?
else
  source venv/bin/activate
  srun python3 "$@"
fi

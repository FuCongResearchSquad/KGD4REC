GPU=${GPU:-"0"}

MODEL_NAME=${MODEL_NAME:-"KGDEncoder"}

DATASET=${DATASET:-"Software"}

REGENERATE=${REGENERATE:-1}
# 1: regenerate dataset cache
# 0: keep previous result

EARLY_STOP=${EARLY_STOP:-4}
NUM_WORKERS=${NUM_WORKERS:-8}

BATCH_SIZE=${BATCH_SIZE:-512}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8192}

COMMON_ARGS="--model_name ${MODEL_NAME} \
             --dataset ${DATASET} \
             --regenerate ${REGENERATE} \
             --batch_size ${BATCH_SIZE} \
             --eval_batch_size ${EVAL_BATCH_SIZE} \
             --early_stop ${EARLY_STOP} \
             --num_workers ${NUM_WORKERS} \
             --verbose 0"

STAGE="pretrain"
LOG_ROOT="logs/${DATASET}/${STAGE}"
MODEL_ROOT="save_model/${DATASET}/${STAGE}"
mkdir -p "${LOG_ROOT}"
mkdir -p "${MODEL_ROOT}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"

RUN_NAME="${DATASET}-KGD_Encoder-${RUN_TS}"

LOG_FILE=${LOG_FILE:-"${LOG_ROOT}/${RUN_NAME}.log"}
MODEL_PATH=${MODEL_PATH:-"${MODEL_ROOT}/${RUN_NAME}.pt"}
printf ">>> log: %s\n" "$LOG_FILE"
printf ">>> model: %s\n" "$MODEL_PATH"

python main.py \
  ${COMMON_ARGS} \
  --gpu ${GPU} \
  --log_file "${LOG_FILE}" \
  --model_path "${MODEL_PATH}" \
  "$@" \
  || exit $?

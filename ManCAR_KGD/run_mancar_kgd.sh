GPU=${GPU:-0}

MODEL_NAME=${MODEL_NAME:-"ManCAR_KGD"}

DATASET=${DATASET:-"Software"}
PRETRAIN_INIT_PATH=${PRETRAIN_INIT_PATH:-"example.pt"}
# Enter the corresponding KGD pretrained weights path

REGENERATE=${REGENERATE:-1}
# 1: regenerate dataset cache
# 0: keep previous result

EARLY_STOP=${EARLY_STOP:-3}
NUM_WORKERS=${NUM_WORKERS:-8}

COMMON_ARGS="--model_name ${MODEL_NAME} \
             --dataset ${DATASET} \
             --pretrained_encoder_path ${PRETRAIN_INIT_PATH} \
             --regenerate ${REGENERATE} \
             --early_stop ${EARLY_STOP} \
             --num_workers ${NUM_WORKERS} \
             --verbose 0"

STAGE="task_train"
LOG_ROOT="logs/${DATASET}/${STAGE}/${MODEL_NAME}"
MODEL_ROOT="save_model/${DATASET}/${STAGE}/${MODEL_NAME}"
mkdir -p "${LOG_ROOT}"
mkdir -p "${MODEL_ROOT}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"

RUN_NAME="${DATASET}-${MODEL_NAME}-${RUN_TS}"

LOG_FILE=${LOG_FILE:-"${LOG_ROOT}/${RUN_NAME}.log"}
MODEL_PATH=${MODEL_PATH:-"${MODEL_ROOT}/${RUN_NAME}.pt"}
printf ">>> log: %s\n" "$LOG_FILE"
printf ">>> model: %s\n" "$MODEL_PATH"

python main.py \
  ${COMMON_ARGS} \
  --gpu "${GPU}" \
  --log_file "${LOG_FILE}" \
  --model_path "${MODEL_PATH}" \
  "$@" \
  || exit $?

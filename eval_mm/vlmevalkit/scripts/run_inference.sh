export PATH=/usr/local/cuda/bin:$PATH

export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1
export timestamp=`date +"%Y%m%d%H%M%S"`
export OLD_VERSION='False'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=$(dirname $SCRIPT_DIR):$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading .env file..."
    export $(cat .env | grep -v '^#' | xargs)
    
    # Echo first few characters to verify loading
    if [ -n "$OPENAI_API_KEY" ]; then
        echo "OPENAI_API_KEY loaded: ${OPENAI_API_KEY:0:8}..."
    else
        echo "WARNING: OPENAI_API_KEY not found in .env"
    fi
    
    if [ -n "$OPENAI_API_BASE" ]; then
        echo "OPENAI_API_BASE loaded: ${OPENAI_API_BASE:0:20}..."
    else
        echo "WARNING: OPENAI_API_BASE not found in .env"
    fi
else
    echo "WARNING: .env file not found"
fi

# gpu consumed
# fp16 17-18G
# int4 7-8G

# model to be used
# Example: MODELNAME=MiniCPM-o-2_6
MODELNAME=$1
# datasets to be tested
# Example: DATALIST=MMMU_DEV_VAL
DATALIST=$2

# run on multi gpus with torchrun command
for DATASET in $DATALIST; do
    echo "Starting inference with model $MODELNAME on dataset $DATASET"
    torchrun --master_port 29500 --nproc_per_node=8 run.py --data $DATASET --model $MODELNAME --mode infer --reuse
    torchrun --master_port 29501 --nproc_per_node=8 run.py --data $DATASET --model $MODELNAME --mode infer --reuse

    # for benchmarks which require gpt for scoring, you need to specify OPENAI_API_BASE and OPENAI_API_KEY in .env file
    if [[ "$DATASET" == *"MMBench_TEST"* ]]; then
        echo "Skipping evaluation for dataset $DATASET"
    else
        echo "Starting evaluation with model $MODELNAME on datasets $DATASET"
        python run.py --data $DATASET --model $MODELNAME --nproc 16 --verbose
    fi
done

# run on single gpu with python command
# python run.py --data $DATALIST --model $MODELNAME --verbose --mode infer
# python run.py --data $DATALIST --model $MODELNAME --verbose --mode infer
# echo "Starting evaluation with model $MODELNAME on datasets $DATASET"
# python run.py --data $DATASET --model $MODELNAME --nproc 16 --verbose

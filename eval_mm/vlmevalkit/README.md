./scripts/run_inference.sh MiniCPM-V-2_6 "MathVista_MINI" all

python run.py --data "MathVista_MINI" --model "MiniCPM-V-2_6" --nproc 16 --verbose --retry 1

uv pip install 'torch-2.2.0+cu118-cp310-cp310-linux_x86_64.whl'
uv pip install 'torchvision-0.17.0+cu118-cp310-cp310-linux_x86_64.whl'
uv pip install 'flash_attn-2.6.3+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
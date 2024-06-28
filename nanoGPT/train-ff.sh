# This script assumes your GPUFS disk image have nanoGPT-ff in the /root/ directory.
cd nanoGPT-ff
python3 train.py config/train_shakespeare_char.py

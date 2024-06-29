# Example commands for ISCA24 tutorial

The below commands assume you are in the base codespace directory: (i.e., /workspaces/gem5-bootcamp-env).  These can be copy and pasted into a bash terminal.

## PyTorch MNIST example command:


/usr/local/bin/gem5-vega gem5/configs/example/gpufs/mi300.py --disk-image /tmp/x86-ubuntu-gpu-ml-isca --kernel ./vmlinux-gpu-ml-isca --no-kvm-perf --app gem5-pytorch/MNIST/kvm-ff/pytorch_qs_mnist.py



## Make the m5term util


pushd gem5/util/term ;  make ; popd


## Mounting disk image command:


mkdir mnt
mount -o loop,offset=$((2048*512)) /tmp/x86-ubuntu-gpu-ml-isca mnt

cp -r gem5-pytorch/nanoGPT/nanoGPT-ff/ mnt/root/

umount mnt



## nanoGPT example command:

build/VEGA_X86/gem5.opt -d tutorial_nanogpt --debug-flags=GPUCommandProc  configs/example/gpufs/mi300.py --disk-image gem5-resources/src/x86-ubuntu-gpu-ml/disk-image/x86-ubuntu-gpu-ml  --kernel gem5-resources/src/x86-ubuntu-gpu-ml/vmlinux-gpu-ml  --app gem5-pytorch/nanoGPT/train-ff.sh --skip-until-gpu-kernel=8 --exit-after-gpu-kernel=9



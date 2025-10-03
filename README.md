J0P7MF - GPU programozas (PEMIK)

# whats needed
(or at least, how i use this repo)
- WIN11
- WSL2 (Debian, but i imagine it works on other distros)
- CUDA compatible graphic card
- CUDA toolkit (https://developer.nvidia.com/cuda-downloads)
- CUDA driver
- make sure nvcc is installed
  - nvcc --version
  - sometime its installed, but not in the path, in this case you can use something like this: 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.zshrc

# compile (example)
if you do not know the shader model capability of your card then run this: nvidia-smi --query-gpu=compute_cap --format=csv

- nvcc -O2 ./1_orai/hello_cuda.cu -arch=sm_89 -o ./1_orai/hello_cuda
- nvcc -O2 ./2_orai/varlenvec.cu -arch=sm_89 -o ./2_orai/varlenvec

Here O2 means optimize for speed, arch means your card's shader model compatibility and -o means the output file.

# 1_orai
- the cpu version was not part of the class, its there for learning purposes

# 2_orai
- basicly the same as last time, but with calculated grid sizes and error handling

# Collab setup
!apt-get update -y

!apt-get install -y cuda-12-4

!sudo rm -rf /usr/local/cuda

!sudo ln -s /usr/local/cuda-12.4 /usr/local/cuda

!export PATH=/usr/local/cuda-12.4/bin:$PATH

!export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

!pip install nvcc4jupyter

%load_ext nvcc4jupyter

# prelude.h

This is the "error handling" code the teacher wrote. Basicly macros that encapsulates code and handles errors, prints it, aborts the program.

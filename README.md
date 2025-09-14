J0P7MF - GPU programozas (PEMIK)

# whats needed
(or at least, how i use this repo)
- WIN11
- WSL2
- CUDA compatible card
- CUDA toolkit
- CUDA driver
- make sure nvcc is installed
  - nvcc --version
  - sometime its installed, but not in the path, in this case you can use something like this: 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.zshrc

# compile (example)
nvcc ./1_orai/hello_cuda.cu -o ./1_orai/hello_cuda

# 1_orai
- the cpu version was not part of the class, its there for learning purposes
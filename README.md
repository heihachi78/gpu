J0P7MF - GPU programozas (PEMIK)

# whats needed
- WIN11
- WSL2
- CUDA compatible card
- CUDA toolkit
- CUDA driver
- make sure nvcc is installed
  - nvcc --version
  - sometime its installed, but not in the path, in this case you can use something like this: 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.zshrc

# compile (example)
nvcc ./1_orai/hello_cuda.cu -o hello_cuda
# GPU Setup Guide for WhisperLiveKit

This guide explains how to properly configure your system for GPU acceleration with WhisperLiveKit.

## Prerequisites

Before building and running WhisperLiveKit with GPU support, ensure you have:

1. **NVIDIA GPU** with CUDA compute capability 3.5 or higher
2. **NVIDIA Driver** compatible with CUDA 12.1 (version 515.00 or higher)
3. **Docker** installed on your system
4. **NVIDIA Container Toolkit** installed

## Checking Your System

### 1. Check GPU Availability
```bash
nvidia-smi
```

This command should show your GPU information and driver version.

### 2. Check CUDA Version
```bash
nvcc --version
```

If this command fails, you may need to install the CUDA toolkit.

## Installing NVIDIA Container Toolkit

If you haven't installed the NVIDIA Container Toolkit, follow these steps:

### Ubuntu/Debian:
```bash
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Update package list
sudo apt-get update

# Install nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker daemon
sudo systemctl restart docker
```

### Other Distributions:
Follow the official guide at: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

## Building and Running with GPU Support

### 1. Build the GPU-enabled Docker Image
```bash
docker build -t whisperlivekit-gpu -f Dockerfile.gpu .
```

### 2. Run with GPU Support
```bash
docker run --gpus all -p 8000:8000 --name whisperlivekit whisperlivekit-gpu
```

### 3. Test with Large Model
```bash
docker run --gpus all -p 8000:8000 --name whisperlivekit whisperlivekit-gpu --model large-v3
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "docker: Error response from daemon: could not select device driver"
This error usually means the NVIDIA Container Toolkit is not installed or not configured properly.

Solution:
- Follow the installation steps above
- Restart Docker daemon: `sudo systemctl restart docker`

#### 2. "CUDA out of memory"
This happens when your GPU doesn't have enough VRAM for the selected model.

Solution:
- Use a smaller model (e.g., `--model medium` instead of `--model large`)
- Close other GPU-intensive applications
- Monitor GPU memory usage with `nvidia-smi`

#### 3. "CUDA driver version is insufficient"
This indicates your NVIDIA driver is too old for the CUDA version in the container.

Solution:
- Update your NVIDIA drivers to version 515.00 or higher
- Alternatively, use an older CUDA base image (less recommended)

#### 4. "No CUDA devices found"
This can happen if Docker doesn't have access to the GPU.

Solution:
- Ensure you're using the `--gpus all` flag when running the container
- Verify NVIDIA Container Toolkit installation
- Check that your user has permissions to access Docker

## Model VRAM Requirements

Different Whisper models require different amounts of GPU memory:

| Model | VRAM Required | Recommended GPU |
|-------|---------------|------------------|
| tiny | ~1GB | GTX 1050 or equivalent |
| base | ~1GB | GTX 1050 or equivalent |
| small | ~2GB | GTX 1060 or equivalent |
| medium | ~5GB | GTX 1070 or equivalent |
| large | ~10GB | RTX 2080 or equivalent |
| large-v3 | ~10GB | RTX 2080 or equivalent |

## Performance Tips

1. **Preload Models**: For production use with multiple users, preload models in memory:
   ```bash
   docker run --gpus all -p 8000:8000 --name whisperlivekit whisperlivekit-gpu --model large-v3 --preloaded-model-count 2
   ```

2. **Use Volume Mounts**: Cache downloaded models between container restarts:
   ```bash
   docker run --gpus all -p 8000:8000 --name whisperlivekit -v whisperlivekit-cache:/root/.cache/huggingface/hub whisperlivekit-gpu
   ```

3. **Monitor Resources**: Use `nvidia-smi` to monitor GPU utilization and memory usage during operation.

## Verification Script

To verify your setup works correctly, run the test script:
```bash
chmod +x test_cuda_compatibility.py
docker build -t cuda-test -f Dockerfile.test .
docker run --gpus all cuda-test
```

This will run a series of tests to verify CUDA functionality.
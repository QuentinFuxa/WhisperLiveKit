# CUDA Compatibility Fixes Summary

## Problem Identified
The Dockerfiles were using CUDA 12.9.1, which is not officially supported by PyTorch. This caused compatibility issues when trying to run with large models.

## Changes Made

### 1. Updated Main Dockerfile
- Changed base image from `nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04` to `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`
- Updated PyTorch installation to use CUDA 12.1 compatible packages
- Changed from `torch torchaudio` to use the cu121 index URL

### 2. Updated GPU Dockerfile
- Changed base image from `nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04` to `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`
- Updated PyTorch installation to use specific CUDA 12.1 compatible versions (2.1.0)
- Both torch and torchaudio now use CUDA 12.1 builds

### 3. Updated README Documentation
- Added CUDA driver requirements
- Specified that CUDA 12.1 is used
- Added note about driver compatibility
- Clarified the correct Dockerfile to use for GPU (`Dockerfile.gpu`)

### 4. Updated pyproject.toml
- Added minimum version requirements for torch (>=2.1.0) and torchaudio (>=2.1.0)
- Ensures compatibility with CUDA 12.1 builds

### 5. Added Supporting Files
- Created `test_cuda_compatibility.py` to verify CUDA setup
- Created `Dockerfile.test` for testing CUDA compatibility
- Created `GPU_SETUP_GUIDE.md` with comprehensive setup instructions

## Why CUDA 12.1?
- Officially supported by current PyTorch versions
- Compatible with widely available NVIDIA drivers (515+)
- Provides good performance for Whisper models
- Balances compatibility with performance

## Testing the Changes
To verify the fixes work:

1. Build the test container:
   ```
   docker build -t cuda-test -f Dockerfile.test .
   ```

2. Run the compatibility test:
   ```
   docker run --gpus all cuda-test
   ```

3. Build the actual GPU container:
   ```
   docker build -t whisperlivekit-gpu -f Dockerfile.gpu .
   ```

4. Run with large model:
   ```
   docker run --gpus all -p 8000:8000 whisperlivekit-gpu --model large-v3
   ```

These changes ensure that WhisperLiveKit will work reliably with GPU acceleration, including support for the large models.
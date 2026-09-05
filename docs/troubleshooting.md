# Troubleshooting


## GPU drivers & cuDNN visibility

### Linux error: `Unable to load libcudnn_ops.so* / cudnnCreateTensorDescriptor`
> Reported in issue #271 (Arch/CachyOS)

`faster-whisper` (used for the SimulStreaming encoder) dynamically loads cuDNN.  
If the runtime cannot find `libcudnn_*`, verify that CUDA and cuDNN match the PyTorch build you installed:

1. **Install CUDA + cuDNN** (Arch/CachyOS example):
   ```bash
   sudo pacman -S cuda cudnn
   sudo ldconfig
   ```
2. **Make sure the shared objects are visible**:
   ```bash
   ls /usr/lib/libcudnn*
   ```
3. **Check what CUDA version PyTorch expects** and match that with the driver you installed:
   ```bash
   python - <<'EOF'
   import torch
   print(torch.version.cuda)
   EOF
   nvcc --version
   ```
4. If you installed CUDA in a non-default location, export `CUDA_HOME` and add `$CUDA_HOME/lib64` to `LD_LIBRARY_PATH`.

Once the CUDA/cuDNN versions match, `whisperlivekit-server` starts normally.

### Windows error: `Could not locate cudnn_ops64_9.dll`
> Reported in issue #286 (Conda on Windows)

PyTorch bundles cuDNN DLLs inside your environment (`<env>\Lib\site-packages\torch\lib`).  
When `ctranslate2` or `faster-whisper` cannot find `cudnn_ops64_9.dll`:

1. Locate the DLL shipped with PyTorch, e.g.
   ```
   E:\conda\envs\WhisperLiveKit\Lib\site-packages\torch\lib\cudnn_ops64_9.dll
   ```
2. Add that directory to your `PATH` **or** copy the `cudnn_*64_9.dll` files into a directory that is already on `PATH` (such as the environment's `Scripts/` folder).
3. Restart the shell before launching `wlk`.

Installing NVIDIA's standalone cuDNN 9.x and pointing `PATH`/`CUDNN_PATH` to it works as well, but is usually not required.

---

## PyTorch / CTranslate2 GPU builds

### `Torch not compiled with CUDA enabled`
> Reported in issue #284

If `torch.zeros(1).cuda()` raises that assertion it means you installed a CPU-only wheel.  
Install the GPU-enabled wheels that match your CUDA toolkit:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

Replace `cu130` with the CUDA version supported by your driver (see [PyTorch install selector](https://pytorch.org/get-started/locally/)).  
Validate with:

```python
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
```

### CTranslate2 does not detect the GPU

The Linux and Windows pip wheels support GPU execution; a zero device count does not by itself mean that you installed a separate CPU-only wheel. See the [CTranslate2 installation documentation](https://opennmt.net/CTranslate2/installation.html) for runtime-library requirements.

Check both runtimes in the environment that launches WLK:

```python
import ctranslate2
import torch

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Torch GPU available:", torch.cuda.is_available())
print("CTranslate2:", ctranslate2.__version__)
print("CTranslate2 GPU count:", ctranslate2.get_cuda_device_count())
```

PyTorch and CTranslate2 load their own libraries. A working Torch CUDA probe does not prove that CTranslate2 can find compatible cuBLAS/cuDNN libraries. Preserve the full error and check the driver, platform, installed wheel, and library search paths before replacing packages.

### Triton or PTXAS rejects a GPU architecture

An error such as `Value 'sm_121a' is not defined for option 'gpu-name'` indicates that the invoked compiler does not recognize the requested architecture. Collect the GPU/driver information from `nvidia-smi`, the CUDA compiler version, and the installed Torch/Triton versions. GPU compute capability and CUDA toolkit version are different identifiers.

Use a compiler and runtime combination supporting that GPU. Avoid copying binaries over files inside an installed Triton package: that creates an environment the lockfile cannot reproduce. Consult the upstream runtime's installation and compatibility documentation for the exact versions involved.

## CPU throughput

### Transcription falls further and further behind on CPU

If the transcript lags more the longer a session runs, the machine is spending
more time on ASR than the stream produces audio. Watch the `Compute` lag shown
in seconds in the web UI: if it keeps increasing during a steady stream, the
backlog is growing faster than the backend can clear it.

Each inference pass costs roughly the same regardless of how much *new* audio it
covers, so when chunks are short most of that work re-encodes audio the previous
pass already saw. `--asr-coalesce-min-s` waits for more new audio before running
a pass, which cuts the number of passes at the cost of updating the transcript
less often:

```bash
wlk --model base --asr-coalesce-min-s 0.75
```

Off by default. Three things worth knowing before you tune it:

- It is most useful with the whisper-family backends, whose processors infer
  once per arriving chunk. The outer gate is available to every backend, but
  backends that already batch internally gain little or nothing and a threshold
  above their own cadence only adds latency.
- The useful value depends on how large the incoming chunks already are, and the
  response is a step rather than a gradual curve: a threshold below the typical
  chunk size does almost nothing, and just above it can halve the passes. Start
  near your chunk size and measure.
- Words reach the screen in larger, less frequent updates, and the first word of
  an utterance arrives later. Held-back audio is bounded by the threshold plus
  one chunk, and is always drained at silences, speaker changes and end of
  stream, so nothing is delayed past a boundary.

If compute time is comfortably below elapsed time and the transcript still lags,
this is not the right fix.

---

Need help with another recurring issue? Open a GitHub discussion or PR and reference this document so we can keep it current.

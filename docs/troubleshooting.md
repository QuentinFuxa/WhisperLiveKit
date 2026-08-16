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
print(torch.cuda.is_available(), torch.cuda.get_device_name())
```

### `CTranslate2 device count: 0` or `Could not infer dtype of ctranslate2._ext.StorageView`
> Follow-up in issue #284

`ctranslate2` publishes separate CPU and CUDA wheels. The default `pip install ctranslate2` brings the CPU build, which makes WhisperLiveKit fall back to CPU tensors and leads to the dtype error above.

1. Uninstall the CPU build: `pip uninstall -y ctranslate2`.
2. Install the CUDA wheel that matches your toolkit (example for CUDA 13.0):
   ```bash
   pip install ctranslate2==4.5.0 -f https://opennmt.net/ctranslate2/whl/cu130
   ```
   (See the [CTranslate2 installation table](https://opennmt.net/CTranslate2/installation.html) for other CUDA versions.)
3. Verify:
   ```python
   import ctranslate2
   print("CUDA devices:", ctranslate2.get_cuda_device_count())
   print("CUDA compute types:", ctranslate2.get_supported_compute_types("cuda", 0))
   ```

**Note for aarch64 systems (e.g., NVIDIA DGX Spark):** Pre-built CUDA wheels may not be available for all CUDA versions on ARM architectures. If the wheel installation fails, you may need to compile CTranslate2 from source with CUDA support enabled.

If you intentionally want CPU inference, run `wlk --backend whisper` to avoid mixing CPU-only CTranslate2 with a GPU Torch build.

---

## Hopper / Blackwell (`sm_121a`) systems
> Reported in issues #276 and #284 (NVIDIA DGX Spark)

CUDA 12.1a GPUs (e.g., NVIDIA GB10 on DGX Spark) ship before some toolchains know about the architecture ID, so Triton/PTXAS need manual configuration.

### Error: `ptxas fatal : Value 'sm_121a' is not defined for option 'gpu-name'`

If you encounter this error after compiling CTranslate2 from source on aarch64 systems, Triton's bundled `ptxas` may not support the `sm_121a` architecture. The solution is to replace Triton's `ptxas` with the system's CUDA `ptxas`:

```bash
# Find your Python environment's Triton directory
python -c "import triton; import os; print(os.path.dirname(triton.__file__))"

# Copy the system ptxas to Triton's backend directory
# Replace <triton_path> with the output above
cp /usr/local/cuda/bin/ptxas <triton_path>/backends/nvidia/bin/ptxas
```

For example, in a virtual environment:
```bash
cp /usr/local/cuda/bin/ptxas ~/wlk/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
```

**Note:** On DGX Spark systems, CUDA is typically already in `PATH` (`/usr/local/cuda/bin`), so explicit `CUDA_HOME` and `PATH` exports may not be necessary. Verify with `which ptxas` before copying.

### Alternative: Environment variable approach

If the above doesn't work, you can try setting environment variables (though this may not resolve the `sm_121a` issue on all systems):

```bash
export CUDA_HOME="/usr/local/cuda-13.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Tell Triton where the new ptxas lives
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"

# Force PyTorch to JIT kernels for all needed architectures
export TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0 12.0 12.1a"
```

After applying the fix, restart `wlk`. Incoming streams will now compile kernels targeting `sm_121a` without crashing.

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

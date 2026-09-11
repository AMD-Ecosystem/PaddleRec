# PaddleRec -- AMD ROCm / AMD Instinct Enablement

This is the AMD ROCm enablement port of [PaddleRec](https://github.com/PaddlePaddle/PaddleRec), delivered on the `port/PaddleRec` branch of [AMD-AIOSS/hipSHIFT](https://github.com/AMD-AIOSS/hipSHIFT).

PaddleRec is a large-scale recommendation framework built on PaddlePaddle. This port validates it on AMD Instinct GPUs (MI300X / gfx942, MI350X-MI355X / gfx950) using ROCm 10.0.0 and a ROCm-enabled PaddlePaddle build.

## Runs on AMD Instinct

| GPU | Architecture | Status |
|---|---|---|
| AMD Instinct MI300X | gfx942 | Validated -- 5/5 passed |
| AMD Instinct MI350X / MI355X | gfx950 | Validated -- 4/4 passed |
| AMD Instinct MI450 (FFM simulator) | gfx1250 | Validated (functional sim) -- 4/4 passed |

## ROCm Install Path

PaddleRec rides PaddlePaddle, which has no public AMD-Instinct wheel on PyPI. The install is source-based:

### 1. Prerequisites

- AMD Instinct GPU (MI300X, MI350X/MI355X, or MI450)
- Ubuntu 24.04, Python 3.11
- AMD GPU driver (amdgpu-dkms); add yourself to render/video groups

```bash
sudo apt install libatomic1 libquadmath0 python3.11 python3.11-venv
sudo usermod -a -G render,video $LOGNAME
# log out and back in (or reboot) to apply group membership
```

### 2. Install ROCm 10.0.0 (public pip wheels)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,devel,device-gfx942]"
# for MI350X/MI355X (gfx950): replace device-gfx942 with device-gfx950
```

### 3. Build PaddlePaddle-on-ROCm from source

There is no public AMD-Instinct PaddlePaddle wheel (the paddlepaddle-rocm PyPI channel targets Hygon DCU, not AMD Instinct). Build from the AMD ROCm fork:

```bash
git clone -b port/Paddle https://github.com/AMD-AIOSS/hipSHIFT.git paddle-rocm && cd paddle-rocm
# follow the build instructions in paddle-rocm/hipshift/release_docs/installation.md
```

### 4. Install PaddleRec from this branch

```bash
# (after building and installing PaddlePaddle-on-ROCm above)
git clone -b port/PaddleRec https://github.com/AMD-AIOSS/hipSHIFT.git paddlerec && cd paddlerec
pip install -r requirements-rocm.txt
pip install .
```

### 5. Verify

```bash
rocminfo | grep -i 'Marketing Name'   # expect: AMD Instinct MI300X (or MI355X)
```

```python
import paddle
print('Compiled with ROCm:', paddle.device.is_compiled_with_rocm())  # expect: True
paddle.set_device('gpu:0')
a = paddle.randn([512, 512])
print('GPU matmul result:', float((a @ a).abs().mean()))  # expect: a positive float
```

Train a DNN ranking model on GPU:

```bash
python -u tools/trainer.py -m models/rank/dnn/config.yaml \
    -o runner.use_gpu=True runner.epochs=3
# Expected: epoch 2 AUC approaching 1.0, RC=0
```

## AMD-specific files in this branch

| File | Purpose |
|---|---|
| `requirements-rocm.txt` | ROCm-compatible requirements (replaces CUDA deps) |
| `requirements-rocm-optional.txt` | Optional ROCm extras (faiss-cpu fallback for ANN retrieval) |
| `amd-constraints.txt` | Version pins for ROCm 10.0.0 + AMD Instinct |
| `Dockerfile.rocm` | ROCm base Dockerfile for AMD Instinct |
| `VALIDATION_NOTES.md` | Per-arch validation results and gotchas |

## Documentation

Full AMD AIOSS release documentation is under `hipshift/release_docs/`:
- [Installation](hipshift/release_docs/installation.md)
- [Compatibility Matrix](hipshift/release_docs/compatibility-matrix.md)
- [Build & Validation Report](hipshift/release_docs/build-validation-report.md)
- [Benchmark Report](hipshift/release_docs/benchmark-report.md)
- [Release Notes](hipshift/release_docs/release-notes.md)

## License

Apache-2.0. See upstream LICENSE. AMD modifications: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

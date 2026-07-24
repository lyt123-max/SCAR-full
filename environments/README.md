# Rebuttal Environments

Each method runs in an independent Conda environment. The YAML files are
reproducible input specifications; `scripts/experiments/freeze_conda_env.py`
must be run on the remote server after creation to save the platform-specific
explicit lock, `pip freeze`, CUDA driver, and GPU UUID.

| Method | Environment file | Notes |
| --- | --- | --- |
| SCAR / PaAno | `scar-paano-cu126.yml` | Formal target: Python 3.11, PyTorch 2.7.1, CUDA 12.6 |
| CATCH | `catch-cu126.yml` | Official CATCH requirements on the same CUDA runtime |
| PGRF-Net | `pgrf-cu126.yml` | Official modules with a project-side path adapter |
| MEMTO | `memto-official.yml` / `memto-cu126-compat.yml` | Try official lock first; use compatibility lock only if the 4090 rejects it |
| PUAD | `puad-official.yml` / `puad-cu126-compat.yml` | Try official lock first; compatibility lock records the minimum CUDA upgrade |

The official and compatibility specifications are intentionally both kept.
Never silently upgrade an old baseline environment. Record which lock was used
in the experiment manifest.

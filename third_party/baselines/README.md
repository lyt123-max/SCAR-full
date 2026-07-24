# Official Baseline Repository Manifest

This directory stores independent upstream Git worktrees and official non-Git
source distributions used for SCAR rebuttal baseline reproduction. The
baseline directories themselves are ignored by the parent project and must not
be edited when implementing SCAR-side data adapters or evaluation wrappers.

Snapshot date: 2026-07-24

| Method | Local path | Upstream | Branch | Pinned commit |
| --- | --- | --- | --- | --- |
| PaAno | `third_party/baselines/PaAno/` | `https://github.com/jinnnju/PaAno.git` | `main` | `d4c67116190efa4592dc6a8a157ced0def68b6af` |
| PUAD | `third_party/baselines/PUAD/` | `https://github.com/LiYuxin321/PUAD.git` | `main` | `41e8b4377e6baa83e56b8f3acdb60ff04ed6c892` |
| PGRF-Net | `third_party/baselines/PGRF-Net/` | `https://github.com/jahoonjeong/PGRF-Net.git` | `main` | `5dc6f7522d20043eb31f6b2b13091c80ad394dcb` |
| GDFlex | `third_party/baselines/GDFlex/` | `https://github.com/makoIM/GDFlex.git` | `main` | `4d93205e7039b08a39995425bb35bee3ec8587a6` |
| MEMTO | `third_party/baselines/MEMTO/` | `https://github.com/gunny97/MEMTO.git` | `main` | `5a3287103021c5c7e7cac9377c626cf18bdea50c` |
| CATCH | `third_party/baselines/CATCH/` | `https://github.com/decisionintelligence/CATCH.git` | `master` | `3647c69be5eb56649b072596cf89098e689e20c3` |

## Source Tree And License Audit

| Method | Git tree | Upstream license file |
| --- | --- | --- |
| PaAno | `b62db1a4dc891bb6264430f4dd775887521a3b10` | `LICENSE` present |
| PUAD | `1779d7b6cef9128b2806090a6d5792f5487764d6` | Not present in pinned upstream |
| PGRF-Net | `26a5b4b84d7192bbc4a5e4de7b2336c6f243ec6d` | Not present in pinned upstream |
| GDFlex | `8e0f892e1abbfb9da27f58b8cb7a5052d72074a7` | `LICENSE` present |
| MEMTO | `218face3c97d4ab2476a054c5fc53a9f297a73d5` | Not present in pinned upstream |
| CATCH | `a3a50afa37d1e8348aa495fdbfe479ace8f63f00` | Not present in pinned upstream |

Absence of a license file is recorded as an audit fact and is not interpreted
as permission to redistribute upstream source.

## Official Non-Git Distribution

DAMP was downloaded from the authors'
[official documentation page](https://sites.google.com/view/discord-aware-matrix-profile/documentation).
The formal SCAR multivariate adaptation must use `DAMP_Multidim.m`; the other
files are retained as the official reference implementation, documentation,
variants, and smoke-test data.

| File | Google Drive file ID | SHA-256 |
| --- | --- | --- |
| `DAMP/DAMP_2_0.m` | `1EPDhFXQ2goTJ5m_x1S84-KNpQFsRNVmf` | `b9e5bed7c07a19892bc40abe1b19045ce8559e4abfcfa5729bfcccdd5584d4b7` |
| `DAMP/DAMP_2.0_How_To_Use.pptx` | `1_-LGilUJpYRbRZpitw05EgkiOZX52kRd` | `5fefcd97eeb89ef5b40fe969ab1c40ba7e3207845b7f56a2da7366d9c8c74d15` |
| `DAMP/BourkeStreetMall.txt` | `1F4Ir-UGOwCBf6i8FXeD3qAzWvEHkUrJC` | `680c7aa32415eccb0661b87fbfff8eb57dbf6866e0df439109a38091841fc283` |
| `DAMP/DAMP_topK.m` | `1SRpPXDB2SHFeXsTP2dLfFHVWyOVpFmP_` | `d83a50259fe47463baa04652db7ef5c5eb40869c562db1776f5d5938bf9766b4` |
| `DAMP/DAMP_fullMP.m` | `1weMlO2u9Wc62TtoRz6Pu2jsOSj7-8rWS` | `bc7930010670a012effe7790adec5f48d3272924a8b7f855c8881f05f9eb5e10` |
| `DAMP/DAMP_X_Lag_Amnesic.m` | `1CqnhApfi_dmMkjBQsDC7m7_2uBqIWskA` | `c131ecbb303e7587cc3ba0fe5bf5ce55c701dda0c1d4079544af4d04f2cdf7dd` |
| `DAMP/DAMP_Golden.m` | `11EA9yxV-pcbB76E8Bna45MIrKrEMmWYo` | `7d53475ae43f08c2a4da91b288feea655ac10ba0e5a393582865c926224acd9c` |
| `DAMP/DAMP_Multidim.m` | `1T5mBxVar3Thyfa0htVUC5jLhx7R9PSow` | `7d9471c901476405b92efea6a4b5e4f9be6069c3523b01cc6cfce7d109272d52` |

The official page labels the standard file as `DAMP 2.0.m`; the local filename
is normalized to `DAMP_2_0.m` to match its MATLAB function declaration. Its
contents and SHA-256 are unchanged.

## Repository Rules

- Treat each baseline directory as read-only upstream code.
- Put SCAR dataset conversion, launch, score alignment, and unified evaluation
  code outside these repositories.
- Use the same SCAR dataset versions, train/validation/test splits,
  preprocessing, and point-wise AUROC/AP evaluator for formal comparisons.
- Record any future upstream update here before changing a checked-out commit.
- DMemAD is not listed because no public official repository has been located.

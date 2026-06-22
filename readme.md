# Bifurcated Integrated Gradients (BIG)

**Trend-Residual Path Decomposition for Time Series Attribution**  
KCC 2026 XAI Workshop

## Overview

BIG decomposes attribution into trend and residual components by routing the IG integration path through **c → T_x → x**. Path additivity of Integrated Gradients guarantees that the two phases yield an exact decomposition: A_T + A_R = F(x) − F(c_M). Built on top of [TIMING (Jang et al., 2025)](https://arxiv.org/abs/2506.05035).

## Requirements

```bash
pip install torch pytorch-lightning captum pykalman tint
```

## Output Directories

| Directory | Description |
|---|---|
| `results_our/` | Main results — Kalman **smoother**, per-position normalization (Tables 3, 4, 6) |
| `results_filter/` | Ablation — Kalman **filter** variant (Table 5) |
| `results_comp/` | Completeness diagnostic — global normalization only (Table 2) |

## Reproducing Results

### Step 1: Generate Attributions

Run `main_td.py` with `--explainers our_td`. Results are saved to `results_our/`.

```bash
python real/main_td.py \
  --explainers our_td \
  --data boiler \        # boiler | PAM | epilepsy | freezer | wafer
  --device cuda:0 \
  --fold 0 \
  --seed 42 \
  --num_segments 50 \
  --min_seg_len 1 \
  --max_seg_len 36 \
  --testbs 200
```

Repeat for all 5 datasets and folds 0–4.

### Step 2: Evaluate — Component Analysis (Table 3)

Reads from `results_our/`, evaluates Trend vs. Residual per dataset.

```bash
python real/main_preserve_td.py \
  --data boiler \
  --device cuda:0 \
  --fold 0 \
  --seed 42 \
  --num_segments 50 \
  --min_seg_len 1 \
  --max_seg_len 36 \
  --output-file results_td.csv \
  --testbs 200
```

### Step 2: Evaluate — Aggregation & Baseline Comparison (Tables 4, 6)

Reads from `results_our/`, compares `|T+R|` vs. `|T|+|R|` vs. TIMING.

```bash
python real/main_preserve.py \
  --data boiler \
  --device cuda:0 \
  --fold 0 \
  --seed 42 \
  --num_segments 50 \
  --min_seg_len 1 \
  --max_seg_len 36 \
  --output-file results_preserve.csv \
  --testbs 200
```

### (Optional) Kalman Filter Ablation (Table 5)

Switch the import in `main_td.py` from `explainers_td` to the filter variant, and change the save directory to `results_filter/`, then re-run Step 1 and Step 2.

---

## Path Completeness Check (Table 2)

> ⚠️ **Normalization switch required before running this step.**

`check_completeness.py`는 sum(T_signed + R_signed) / (F(x) − F(c))가 1에 수렴하는지 검산하는 코드입니다.  
Per-position normalization은 cross-position 비교용이므로 completeness ratio 계산에 적합하지 않습니다. 실행 전 `explainers_td.py`의 `_ig_phase` 내 normalization을 **global normalization**으로 교체하고, 결과를 `results_comp/`에 저장해야 합니다.

### Completeness check 설정

`_ig_phase` 내부에서 아래 코드를 활성화합니다.

```python
attr = attr_sum / n_alphas
attr = attr.mean(dim=0)
```

그리고 per-position normalization 코드는 주석 처리합니다.

```python
# N_free = time_mask.sum(dim=0)
# attr = attr_sum.sum(dim=0) / (n_alphas * N_free.clamp_min(1))
# attr = torch.where(N_free > 0, attr, torch.zeros_like(attr))
```

또한 `main_td.py` 저장 경로를 `results_comp/`로 변경한 뒤 Step 1을 재실행합니다.

### Main faithfulness evaluation 설정

메인 평가(Tables 3, 4, 6)에서는 반드시 **per-position normalization**을 사용합니다.

```python
N_free = time_mask.sum(dim=0)
attr = attr_sum.sum(dim=0) / (n_alphas * N_free.clamp_min(1))
attr = torch.where(N_free > 0, attr, torch.zeros_like(attr))
```

### Running the diagnostic

```bash
python check_completeness.py \
  --model state \
  --seed 42 \
  --folds 0 1 2 3 4 \
  --results-dir ./results_comp \
  > completeness_check.txt
```

결과(`completeness_check.txt`)에서 `all_med`가 1.0에 가깝고 `norm_err`가 0에 가까우면 정상입니다.

---

## Citation

```bibtex
@inproceedings{lim2026big,
  title     = {Bifurcated Integrated Gradients: Trend-Residual Path Decomposition for Time Series Attribution},
  author    = {Lim, Jiyu and Yeo, Jisu and Lim, Haksoo and Choi, Jaesik},
  booktitle = {KCC 2026 XAI Workshop},
  year      = {2026}
}
```

---



# TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation

## Introduction
Official implementation for **TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation**. TIMING is implemented in PyTorch and tested on different time series datasets, including switch-feature, state, Mimic-III, PAM, Epilespy, boiler, freezer, and wafer. Our overall experiments are based on [time_interpret](https://github.com/josephenguehard/time_interpret), [ContraLSP](https://github.com/zichuan-liu/ContraLSP), [TimeX++](https://github.com/zichuan-liu/TimeXplusplus), [WinIT](https://github.com/layer6ai-labs/WinIT). 
Sincere thanks to each of the original authors!



## Installation instructions

```shell script
conda create -n timing python==3.10.16
conda activate timing
pip install -r requirement.txt --no-deps
```
The requirements.txt file is used to install the necessary packages into a virtual environment.

To test with switch-feature, additional setup is required.

```shell script
git clone https://github.com/TimeSynth/TimeSynth.git
cd TimeSynth
python setup.py install
cd ..
python synthetic/switchstate/switchgenerator.py
```

### MIMIC-III Dataset Setup

To use MIMIC-III dataset, you need to download the original dataset and set up a PostgreSQL database.

**1. Download MIMIC-III Dataset**

- Download from [PhysioNet](https://physionet.org/content/mimiciii/1.4/) (requires credentialed access and CITI training)
- Sign the Data Use Agreement (DUA) and download all CSV files

**2. Load into PostgreSQL**

- Clone the MIMIC-III code repository:
```shell script
git clone https://github.com/MIT-LCP/mimic-code.git
cd mimic-code/mimic-iii/buildmimic/postgres
```
- Follow the repository instructions to create database `mimic` with schema `mimiciii` and load CSV files

**3. Configure Database Connection**

Edit `datasets/mimic3.py` (around lines 151-157) to set your PostgreSQL connection:
```python
dbname = "mimic"
schema_name = "mimiciii"
# Set host, user, and password as needed
```



## Reproducing experiments

We have divided our experiments into two categories: Synthetic and Real.

All experiments can be executed using scripts located in scripts/real, scripts/hmm, or scripts/switchfeature.

This is an example execution for MIMIC-III (ours)
```shell script
bash scripts/real/train.sh
bash scripts/real/run_mimic_our.sh
bash scripts/real/run_mimic_baseline.sh
```

Due to differences between our training environment and the released code, the paper’s results may not be fully reproducible with the training scripts alone. All XAI evaluations were conducted on a single NVIDIA RTX 3090 GPU (24GB VRAM). We have publicly released [all model checkpoints](https://drive.google.com/file/d/1nX9x2iBTcFUykHsKAj_1xQftWvZtc1B6/view?usp=sharing), except those trained on the restricted-access MIMIC-III dataset.

All results will be stored in the current working directory.

And then save parsing results:
```shell script
python real/parse.py --model state --data mimic3 --top_value 100
python real/parse.py --model state --data mimic3 --experiment_name baseline --top_value 100
```

All parsed results will be saved in the results/ directory.

## Citation
```
@inproceedings{jang2025timing,
  title={{TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation}},
  author={Jang, Hyeongwon and Kim, Changhun and Yang, Eunho},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```


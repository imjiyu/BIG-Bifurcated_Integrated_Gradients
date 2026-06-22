# Bifurcated Integrated Gradients (BIG)
Trend-Residual Path Decomposition for Time Series Attribution
KCC 2026 XAI Workshop

## Overview
BIG decomposes attribution into trend and residual components
by routing the IG path through c → T_x → x.
Built on top of TIMING (Jang et al., 2025).

## Requirements
...

## Output Directories
...

## Reproducing Results
Step 1: Attribution 생성 (main_td.py)  
Step 2: 평가 (main_preserve_td.py / main_preserve.py)  

## Path Completeness Check (Table 2)  
> ⚠️ **Normalization switch required**

`check_completeness.py`는 Path Completeness를 확인하기 위한 코드입니다.  
따라서 실행 전, 초기 attribution 값을 구할 때 `explainers_td.py`의 `_ig_phase` 함수 내부에서 attribution normalization 방식을 **per-position normalization**에서 **global normalization**으로 변경해야 합니다.

### Completeness check 설정  

`_ig_phase` 내부에서 아래 코드를 사용합니다.

```python
attr = attr_sum / n_alphas
attr = attr.mean(dim=0)
```

그리고 기존 per-position normalization 코드는 주석 처리합니다.

```python
# N_free = time_mask.sum(dim=0)
# attr = attr_sum.sum(dim=0) / (n_alphas * N_free.clamp_min(1))
# attr = torch.where(N_free > 0, attr, torch.zeros_like(attr))
```

### Main faithfulness evaluation 설정

메인 faithfulness 평가는 기존 방식인 **per-position normalization**을 사용합니다.

```python
N_free = time_mask.sum(dim=0)
attr = attr_sum.sum(dim=0) / (n_alphas * N_free.clamp_min(1))
attr = torch.where(N_free > 0, attr, torch.zeros_like(attr))
```

즉, `check_completeness.py`를 실행할 때만 global normalization으로 변경하고, 일반적인 faithfulness 평가에서는 per-position normalization을 사용합니다.


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


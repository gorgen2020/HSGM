# LATENT-TO-OBSERVABLE SCORE CORRECTION FOR PROBABILISTIC TIME SERIES IMPUTATION

This repository contains the implementation of **Latent-to-Observable Score Correction** for probabilistic time series imputation.

---

## 📂 File Structure & Debug Steps  
Example below uses the **Synthetic** dataset:

1. **train_vae**  
   Pre-train the VAE architecture.

2. **train_vada**  
   Train the latent diffusion model.

3. **Run Final Execution**  
   Execute the script below to obtain the final imputation results:  
   ```
   python exe_Synthetic.py
   ```

---

## 📊 Data Availability  

| Dataset | Source | Preprocessing |
|---------|-------|----------------|
| **P2012** | [CSDI GitHub](https://github.com/ermongroup/CSDI/tree/main) | Follow the instructions provided in the repository. |
| **ETT** | [SAITS GitHub](https://github.com/WenjieDu/SAITS/tree/main) | Download and preprocess as described in the repo. |
| **MIMIC-IV** | [PhysioNet](https://physionet.org/content/mimiciv/3.1/) | Obtain the raw data and preprocess using [MedFuse](https://github.com/nyuad-cai/MedFuse/tree/main). |

---

## ⚡ Quick Start Example
```bash
# 1. Pre-train VAE
python train_vae.py --dataset Synthetic

# 2. Train latent diffusion
python train_vada.py --dataset Synthetic

# 3. Run observation imputation
python exe_Synthetic.py
```


---
## 📌 Notes
- Ensure all required dependencies are installed before running the scripts.  
- Modify dataset paths in the scripts if your directory structure differs.  

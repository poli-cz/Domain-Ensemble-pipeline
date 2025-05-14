# Deep Domain Detection

This repository contains the implementation and experiments from the master's thesis:

**Porovnání klasifikačních metod pro účely detekce maligních domén**  
*Comparison of Classification Methods for Malicious Domain Detection*  
Author: Bc. Jan Polišenský  
Supervisor: Ing. Radek Hranický, Ph.D.  
Faculty of Information Technology, Brno University of Technology (FIT VUT)  
Academic Year: 2024/25

---

## 🧠 Project Overview

This project focuses on detecting malicious domains using machine learning methods and compares the performance of various classifiers, including neural networks, support vector machines (SVM), and tree-based algorithms.

The main contribution is a **multi-stage classification pipeline** with a **decision meta-model** and an optional **false-positive detection module**. The solution achieved:

- **Macro-F1 Score**: 0.984  
- **F1 Score for Phishing**: 0.985  
- **F1 Score for Malware**: 0.980

Key elements of the pipeline include:
- **176-dimensional feature vector** combining data from TLS, DNS, RDAP, GeoIP, and lexical analysis.
- Extensive experimental validation against independent verification datasets.
- Robust ensemble design based on model stacking and classifier diversity.

## 🛠️ Technical Details

---

The classification system is designed to support multiple **stages** of prediction based on the availability of input features. This allows the model to function effectively even when only partial data is available during inference.

### 🔢 Feature Stages

- **Stage 1** – *62 features*  
  Lexical-only features, such as character distributions, domain length, entropy, digit ratios, etc.

- **Stage 2** – *128 features*  
  Includes Stage 1 features, plus enriched data from:
  - DNS queries
  - IP address metadata
  - RDAP/WHOIS fields

- **Stage 3** – *176 features*  
  Full feature vector including:
  - TLS certificate fields
  - GeoIP information
  - Aggregated HTML and lexical signals

### 🧩 System Components

- **Baseline Models**  
  Each stage is associated with trained classifiers, including:
  - Feedforward Neural Networks (FFNN)
  - Convolutional Neural Networks (CNN)
  - LightGBM
  - XGBoost
  - Support Vector Machines (SVM)

- **Meta-Model**  
  A fully connected neural network trained on predictions from the baseline models to produce a final decision.

- **False Positive Detector (FPD)**  
  A lightweight secondary model that detects and filters likely false positives from the final prediction, further improving precision.

### ⚙️ Dynamic Inference Logic

The pipeline automatically detects which stage of features is available and uses the appropriate submodels. This enables adaptive classification in environments where some data sources may be missing or delayed (e.g., no TLS or GeoIP yet available).

Model versioning and stage logic can be manually controlled by editing:

```
src/pipeline.py
```

## 📦 Project Structure

```
deep_domain_detection/
├── docs/                 # Thesis sources, visualizations, confusion matrices
│   ├── tex_sources/      # LaTeX source files of the thesis
│   └── figures/          # Plots, SHAP images, architecture diagrams
├── experiments/          # SHAP analysis, model comparison, feature testing
├── src/                  # All source code
│   ├── core/             # Core logic: loaders, transformers, pipeline
│   ├── models/           # Saved model files (.keras, .pkl, .xgb, etc.)
│   ├── training/         # Training scripts as Jupyter notebooks
│   ├── ensemble_pipeline_example.ipynb # Full system usage example
├── poetry.lock           # Locked dependency versions
├── pyproject.toml        # Poetry project definition
└── README.md             # You are here
```

Each subfolder contains more detailed documentation or examples relevant to its content.

---

## 🛠️ Installation & Setup

> ⚠️ **Hardware Requirements Notice**  
> - **Training** deep neural models (CNN, FFNN, meta-model) requires a GPU with **at least 8 GB of VRAM**.  
> - **Inference** can be run on CPU, but it is significantly slower.  
> - If you plan to use CPU-only, you **do not need to install CUDA-related packages** (e.g., `torch-cuda`, `tensorflow-gpu`).


### 1. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Clone and install dependencies

```bash
git clone https://github.com/your-username/deep_domain_detection.git
cd deep_domain_detection
poetry install
```

### 3. Activate the virtual environment

```bash
poetry shell
```

### 4. Run Notebooks with Correct Interpreter

- Open VS Code and select the Poetry environment as the Python interpreter
- Or run Jupyter:

```bash
poetry run jupyter notebook
```

---

## 🚀 Getting Started

To see the pipeline in action, open the following notebook:

```
src/ensemble_pipeline_example.ipynb
```

This notebook shows how to:
- Load a trained ensemble pipeline
- Classify new domain samples
- (Optionally) generate SHAP explanations for interpretability

Training code for individual models is available in:

```
src/training/
```

---

- This repository is distributed with already pretrained models.  
- Datasets are large and are **not included** directly in this repository.  
  Download them separately and place them into the following directory:

```
src/parkets/
```

If you want to retrain the models from scratch, follow these steps:

### 🔁 Model Training Workflow

1. **Download datasets**  
   Use the official [DomainRadar repository](https://github.com/nesfit/domainradar-clf/tree/main/testdata) to get the training datasets. Place the required `.parquet` files (e.g. benign and malicious domains) into:

   ```
   src/parkets/
   ```

2. **Select a training notebook**  
   Each model has a corresponding Jupyter notebook for training:

   ```
   src/training/feedforward_train.ipynb
   src/training/cnn_train.ipynb
   src/training/xgb_lgbm_train.ipynb
   ...
   ```

3. **Update dataset paths and labels**  
   Open the training notebook of your choice and set the correct path to the dataset and label name.

4. **Train baseline models**  
   Train individual classifiers such as:
   - Feedforward Neural Network (FFNN)
   - Convolutional Neural Network (CNN)
   - LightGBM (LGBM)
   - XGBoost (XGBC)

5. **Train meta-model**  
   After training base classifiers, use:

   ```
   src/training/meta_model_train.ipynb
   ```

   to train the ensemble decision-level meta-classifier.

6. **Train false-positive detector (FPD)**  
   Still within `meta_model_train.ipynb`, configure and train the FPD module, which reduces false alarms.

---

### 🔧 Customizing the Final Pipeline

The final classification pipeline supports configurable model versions.

To change which models are used in the final ensemble, edit the version selection logic in:

```
src/pipeline.py
```

You can define different model sets for each classification stage or threat type (e.g., phishing vs. malware).

---

## 📖 Thesis Citation

> POLIŠENSKÝ, Bc. Jan. *Porovnání klasifikačních metod pro účely detekce maligních domén.* Brno, 2024. Diplomová práce. Vysoké učení technické v Brně, Fakulta informačních technologií. Vedoucí práce Ing. Radek Hranický, Ph.D.

---

## 🔑 Keywords

`malicious domains` · `phishing` · `malware` · `machine learning` · `neural networks` · `SVM` · `cybersecurity`

---

## 📜 License

Academic and non-commercial use only. For other uses, please contact the author.

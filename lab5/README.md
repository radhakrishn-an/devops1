# MLOps CI/CD --- GitHub Actions + Hugging Face Hub

Every push to `main` trains a model, evaluates it, and --- only if it
passes an accuracy threshold --- deploys it to Hugging Face Hub
automatically.

## How it works

``` text
Push to GitHub
      │
      ▼
   ┌──────────┐
   │   test   │
   │  pytest  │
   └────┬─────┘
        │
        ▼
┌──────────────────────┐
│ train_and_evaluate   │
│ prepare → train →    │
│ evaluate             │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────┐
│ deploy_to_huggingface   │
│ only on main + gate     │
│ passed                  │
└─────────────────────────┘
```

## Dataset

This project uses the Boston Housing dataset.

The original Boston Housing dataset is a regression dataset. For this
project, the target is converted into a binary classification problem
using the median house price.

-   `0` → house price is less than or equal to the median
-   `1` → house price is greater than the median

The dataset is divided into:

-   80% training data
-   20% testing data

## Machine Learning Model

The project uses a Random Forest Classifier.

Model parameters are stored in `params.yaml`.

``` yaml
train:
  n_estimators: 100
  max_depth: 10
  random_state: 42
```

## Model Evaluation

The following metrics are calculated:

-   Accuracy
-   Precision
-   Recall
-   F1 Score

The quality gate is:

``` yaml
evaluate:
  min_accuracy: 0.80
```

If the model accuracy is below `0.80`, the pipeline fails and the model
is not deployed.

## Project Structure

``` text
lab5/
├── .github/
│   └── workflows/
│       └── ci-cd.yaml
├── src/
│   ├── __init__.py
│   ├── prepare.py
│   ├── train.py
│   ├── evaluate.py
│   └── register.py
├── tests/
│   └── test_pipeline.py
├── data/
│   ├── train.csv
│   └── test.csv
├── model/
│   ├── model.joblib
│   └── features.json
├── metrics.json
├── params.yaml
├── requirements.txt
└── README.md
```

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Joblib
-   PyYAML
-   Pytest
-   Git
-   GitHub
-   GitHub Actions
-   Hugging Face Hub

## Setup

### Create a virtual environment

``` bash
python3 -m venv .venv
```

### Activate the virtual environment

``` bash
source .venv/bin/activate
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

## Run the Pipeline Locally

### Prepare the dataset

``` bash
python src/prepare.py
```

### Train the model

``` bash
python src/train.py
```

### Evaluate the model

``` bash
python src/evaluate.py
```

### View metrics

``` bash
cat metrics.json
```

### Run tests

``` bash
pytest
```

## GitHub Actions CI/CD

The workflow file is:

`.github/workflows/ci-cd.yaml`

Every push to the `main` branch triggers the pipeline automatically.

The pipeline contains three jobs.

### 1. Test

Runs the pytest sanity checks.

``` bash
pytest
```

### 2. Train and Evaluate

The pipeline performs:

``` text
prepare
   ↓
train
   ↓
evaluate
   ↓
quality gate
```

If the accuracy is below the threshold in `params.yaml`, the pipeline
stops.

### 3. Deploy to Hugging Face

Deployment happens only when:

-   The pipeline runs from `main`
-   Tests pass
-   Training succeeds
-   Evaluation succeeds
-   The model passes the quality gate

The following files are uploaded:

``` text
model.joblib
features.json
README.md
```

## Hugging Face Setup

Hugging Face:

https://huggingface.co/

Hugging Face Access Tokens:

https://huggingface.co/settings/tokens

Create a token with **Write** permission.

Do not put the token inside the source code.

## GitHub Secrets and Variables

Open the GitHub repository:

`Settings → Secrets and variables → Actions`

### Repository Secret

Name:

`HF_TOKEN`

Value:

Your Hugging Face Write Access Token

### Repository Variable

Name:

`HF_REPO_ID`

Value:

`your-huggingface-username/boston-random-forest`

Example:

`username/boston-random-forest`

## GitHub Commands

The project is part of the existing DevOps repository.

From the main `DevOps` repository:

### Check status

``` bash
git status
```

### Add Lab 5

``` bash
git add lab5
```

### Commit

``` bash
git commit -m "Add Lab 5 MLOps CI/CD pipeline"
```

### Push to main

``` bash
git push origin main
```

## Check GitHub Actions

After pushing to `main`:

``` text
GitHub Repository
      ↓
Actions
      ↓
MLOps CI/CD Pipeline
```

The workflow will run:

``` text
Test
  ↓
Train and Evaluate
  ↓
Deploy to Hugging Face
```

## Quality Gate

The quality gate is controlled using:

``` yaml
evaluate:
  min_accuracy: 0.80
```

If:

``` text
Accuracy >= 0.80
```

the model passes and deployment continues.

If:

``` text
Accuracy < 0.80
```

the pipeline fails and the model is not deployed.

## Final Result

``` text
Boston Dataset
      ↓
Data Preparation
      ↓
Train Random Forest
      ↓
Model Evaluation
      ↓
Accuracy Quality Gate
      ↓
GitHub Actions
      ↓
Hugging Face Hub
```

This project demonstrates an end-to-end MLOps CI/CD pipeline for
automated model testing, training, evaluation, quality control, and
deployment.

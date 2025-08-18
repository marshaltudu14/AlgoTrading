# HRM Training on Kaggle/Colab

This directory contains everything you need to train your Hierarchical Reasoning Model (HRM) on cloud platforms using their computational resources.

## Files Included

1. `hrm_training_notebook.ipynb` - Main Jupyter notebook for training
2. `colab_requirements.txt` - Simplified requirements for cloud environments
3. Your complete codebase (when uploaded)

## Setup Instructions

### For Google Colab:

1. **Upload Codebase**:
   - Compress your entire AlgoTrading directory
   - Upload to Google Drive
   - Mount Drive in Colab: `drive.mount('/content/drive')`

2. **Set Project Path**:
   ```python
   PROJECT_PATH = '/content/drive/MyDrive/AlgoTrading'
   ```

3. **Install Dependencies**:
   ```bash
   !pip install -r "/content/drive/MyDrive/AlgoTrading/colab_requirements.txt"
   ```

4. **Upload Data**:
   - Upload your `data/final` directory to Drive
   - Ensure it's in the correct location relative to your project

### For Kaggle:

1. **Create Dataset**:
   - Upload your codebase as a Kaggle dataset
   - Ensure directory structure is preserved

2. **Set Project Path**:
   ```python
   PROJECT_PATH = '/kaggle/input/your-dataset/AlgoTrading'
   ```

3. **Add Dataset to Notebook**:
   - In Kaggle notebook, add your dataset in the "Add data" section

## Training Options

### Quick Test Run:
```python
run_hrm_training(episodes=5, testing_mode=True)
```

### Full Training:
```python
run_hrm_training(episodes=2000, testing_mode=False)
```

### Custom Training:
```python
symbols_to_train = ["Nifty_5", "Bank_Nifty_15", "RELIANCE_1"]
run_hrm_training(episodes=1000, testing_mode=False, symbols=symbols_to_train)
```

## Performance Benefits

1. **GPU Acceleration**: 5-10x faster training on cloud GPUs
2. **More Resources**: Higher RAM and better CPUs
3. **Parallel Processing**: Better utilization of cloud resources
4. **Cost Effective**: Pay only for compute time needed

## Expected Results

With your 13M+ data points across 55 instruments and 10 timeframes:
- 2000 episodes × 5000 steps = 10M training steps
- Coverage: ~76% of your dataset (with early termination)
- Model Size: FIXED at ~5.6M parameters
- Training Time: 2-4 hours on GPU (vs 8-16 hours locally)
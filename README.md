# SMOTENC-XGBoost-Expert

This repository implements the methodology described in the paper "Application of XGBoost and SMOTENC in Food Safety Evaluation Based on Virtual Samples".

## Quick Start

This project consists of two main Python scripts: `SMOTENC_.py` and `XGBoost_Expert.py`.

### SMOTENC_.py

This script is designed to address the issue of class imbalance in the original data by utilizing the SMOTENC algorithm. By applying this algorithm, a more balanced dataset is created, which can improve the performance of subsequent machine learning models.

### XGBoost_Expert.py

In this script, a combination of the XGBoost algorithm and Expert scoring is used to adjust the weights. The XGBoost algorithm is a powerful machine learning algorithm known for its high performance in many classification and regression tasks. By integrating Expert scoring simultaneously, the weight can be adjusted.

## Usage

To use these scripts, make sure you have the required Python libraries installed. The scripts can be run independently, but it is recommended to understand the flow and dependencies between them for a comprehensive analysis.

1. **For `SMOTENC_.py`**: Simply run the script with your input data to obtain a balanced dataset.
2. **For `XGBoost_Expert.py`**: Ensure that the data is preprocessed appropriately before running the script. The script will train an XGBoost model and adjust the weights based on Expert scoring.

## Dependencies

- Python version 3.9
- Relevant Python libraries such as `pandas`, `numpy`, `xgboost`, etc.

## Contributing

If you find any issues or have suggestions for improvement, feel free to contribute by submitting pull requests or opening issues.

## License

This project is licensed under the MIT License.
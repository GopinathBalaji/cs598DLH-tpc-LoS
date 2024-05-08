# cs598DLH-tpc-LoS

To load TPC-LoS, TPC-multitask, LSTM, and Transformer model, navigate to bdeleon2 and start a local runtime notebook. You can connect to our ipynb on colab with the notebook (https://research.google.com/colaboratory/local-runtimes.html) and run the models. Must be on Python 3.6 and meet the `requirements.txt`. Make sure the paths.json is set correctly to where your pre-processed data is. Currently only trained on MIMIC. Pre-processing instructions can be found in bdeleon2 in their respective folders.

If running the limited tests, insert the test files into the test directory inside whichever path you set up the MIMIC data on.

# Length of Stay Prediction using Temporal Pointwise Convolution (TPC)

This repository contains the code and resources for predicting the Length of Stay (LoS) in Intensive Care Units (ICUs) using the Temporal Pointwise Convolution (TPC) model, as well as other baseline models such as LSTM and Transformer.

## Table of Contents

- [cs598DLH-tpc-LoS](#cs598dlh-tpc-los)
- [Length of Stay Prediction using Temporal Pointwise Convolution (TPC)](#length-of-stay-prediction-using-temporal-pointwise-convolution-tpc)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Data Preprocessing](#data-preprocessing)
    - [MIMIC-IV Dataset](#mimic-iv-dataset)
    - [eICU Dataset](#eicu-dataset)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Results](#results)
  - [License](#license)

## Setup

### Prerequisites

- Python 3.6
- PostgreSQL
- Windows machine (for the provided setup guide)

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/cs598DLH-tpc-LoS.git
   cd cs598DLH-tpc-LoS
   ```

2. Install the required Python packages:

   ```
   pip install -r requirements.txt
   ```

3. Set up the PostgreSQL database by following the Length of Stay Prediction Task Setup Guide.

4. Update the `Tpc-LoS-Prediction/paths.json` file with the correct paths to your preprocessed data.

## Data Preprocessing

### MIMIC-IV Dataset

1. Navigate to the MIMIC-IV preprocessing directory:

   ```
   cd Tpc-Los-Prediction/MIMIC-IV_preprocessing
   ```

2. Follow the preprocessing instructions provided in the directory's README.

### eICU Dataset

1. Navigate to the eICU preprocessing directory:

   ```
   cd Tpc-LoS-Prediction/eICU_preprocessing
   ```

2. Follow the preprocessing instructions provided in the directory's README.

## Model Training and Evaluation

1. Open the `DL4H_Team_82.ipynb` notebook in Google Colab.

2. Connect the notebook to a local runtime by following the instructions in the Colab Local Runtimes guide.

3. Run the notebook cells in sequence to train and evaluate the TPC-LoS, TPC-multitask, LSTM, and Transformer models.

4. If running the limited tests, insert the test files into the test directory inside the path where you set up the MIMIC data.

## Results

The experimental results and analysis can be found in the results directory.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For any questions or issues, please open an issue on the GitHub repository.

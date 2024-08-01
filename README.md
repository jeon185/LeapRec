# LeapRec
This project is a PyTorch implementation of LeapRec (CIKM 2024).

The implementation is based on Python 3.10.0 and PyTorch 2.1.2.
A complete list of required packages can be found in the `requirements.txt` file.
Please install the necessary packages before running the code.

## Datasets
We use four datasets in our work: ML-1M, Goodreads, Grocery, and Steam.
The preprocessed dataset is included in the repository: `./data`.

## Source codes
Source codes are included in `./src`.
The codes are divided into two repositories, backbone and reranking.
They are respectively for training a backbone model and reranking the results.

## Pretrained backbone models
In `./out`, we provide pretrained backbone models which are trained under our proposed calibration-aware learning to rank.
Therefore, you can run reranking algorithm without training the model.

## Running the code
To run the model training code, use the `python main.py` with an argument `--data` in the repository `./backbone`.
To run the reranking code, use the `python main.py` with arguments `--data`, `--algorithm`, and `--balance` in the repository `./reranking`.
The argument `--balance` is `lambda` of our paper.
For convenience, we provide a `demo.sh` script that reproduces the experiments presented in our paper.

To customize the configuration, please edit the `config.yaml` file in each repository.
Please refer to our paper if you need guidance on setting the hyperparameters.

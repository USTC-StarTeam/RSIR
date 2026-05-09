# CAN RECOMMENDER SYSTEMS TEACH THEMSELVES? A RECURSIVE SELF-IMPROVING FRAMEWORK WITH FIDELITY CONTROL

This repository contains the code for an iterative method that alternates between training a recommendation model and generating data to improve its performance.

## Usage

The main script for running the process is `run.py`. The behavior of the script is controlled by the `--mode` and `--trainfile` arguments.

### Step 1: Train Recommendation Model 📈

To train the recommendation model, you must set the `--mode` to `'recommendation'` and the `--trainfile` argument to `""`. This tells the script to use the base training file (e.g., `train.pth`).

**Command:**

```bash
# Example for the 'amazon-toys' dataset using the SASRec model
python run.py -m SASRec -d "amazon-toys" --trainfile "" --mode 'recommendation'
```

### Step 2: Generate Augmented Data 📝

After training, you can generate the augmented data for the next iteration. For this step, set the `--mode` to `'generation'` and specify the output file suffix with the `--trainfile` argument.

For example, to generate the data for the first iteration (`_1th`), you would run:

**Command:**

```bash
# Example for generating the 1st iteration data for 'amazon-toys'
python run.py -m SASRec -d "amazon-toys" --trainfile "_1th" --mode 'generation'
```

### Iterative Process 🔁

To improve the model, simply repeat **Step 1** and **Step 2**. For the second iteration, you would first train on the newly generated `train_1th.pth` data (by setting `--trainfile "_1th"` and `--mode 'recommendation'`), and then generate the `train_2th.pth` data (by setting `--trainfile "_2th"` and `--mode 'generation'`).

-----

## Reproducing Paper Results

We provide the augmented data files generated after each iteration, as mentioned in our paper. You can use these files to directly run the recommendation step and verify the reported results without needing to run the data generation steps yourself.

### Provided Data Files

The pre-generated data is available for the following datasets:

  * **amazon-sport:** 8 iterations (`train_1th.pth` to `train_8th.pth`)
  * **yelp:** 8 iterations (`train_1th.pth` to `train_8th.pth`)
  * **amazon-toys:** 1 iteration (`train_1th.pth`)
  * **amazon-beauty:** 1 iteration (`train_1th.pth`)

### Verification Example

To verify the results using a pre-generated data file, run the recommendation command (Step 1) but set the `--trainfile` argument to point to the specific iteration file you want to test.

For example, to test the model's performance on the **8th** iteration for the `amazon-sport` dataset, you would run the following command:

```bash
# Verify results on the 8th iteration data for the 'amazon-sport' dataset
python run.py -m SASRec -d "amazon-sport" --trainfile "_8th" --mode 'recommendation'
```

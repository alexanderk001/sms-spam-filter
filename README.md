# 📩 SMS Spam Filter using Naive Bayes

This repository contains an R implementation of a Naive Bayes classifier for detecting SMS spam messages. The classifier uses text preprocessing techniques and a Document-Term Matrix (DTM) to train and evaluate the model. The dataset used is the publicly available SMS Spam Collection dataset from the UCI Machine Learning Repository.

## Project Overview
The goal of this project is to create a machine learning model that classifies SMS messages as either "ham" (non-spam) or "spam." The implementation uses the Naive Bayes algorithm, which is well-suited for text classification tasks.

## Dataset
The dataset used for this project is the **SMS Spam Collection**, which can be downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection). It contains 5,572 labeled SMS messages with the following columns:

- **Label**: `ham` for non-spam messages and `spam` for spam messages.
- **Message**: The SMS text message.

Save the dataset as `data/sms_spam.txt` in the repository folder.

## Installation and Requirements
### Prerequisites
- R (version 4.0 or later)
- RStudio (optional but recommended)

### Required Libraries
Install the following R libraries:
```R
install.packages(c("dplyr", "readr",  "caret", "e1071", "tm"))
```

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/alexanderk001/sms-spam-filter.git
   cd sms-spam-filter
   ```

2. Place the dataset in the `data/` directory as `sms_spam.txt`.

3. Open the R script in RStudio and run it step by step.

4. Results, including accuracy and confusion matrix, will be printed in the console.

## Code Explanation

### 1. Loading Data
The dataset is loaded using the `readr` package, and the labels (`ham`/`spam`) are converted into factors for modeling.

### 2. Text Preprocessing
The text data is cleaned using the following steps:
- Convert to lowercase.
- Remove punctuation and numbers.
- Remove stop words.
- Strip extra whitespace.

### 3. Document-Term Matrix (DTM)
The cleaned messages are converted into a DTM, which represents the frequency of terms in each document (message). Sparsity is reduced by keeping only frequently used terms.

### 4. Binary Conversion
The DTM is converted to binary values:
- `1` if a term is present.
- `0` if a term is absent.

### 5. Training and Testing
The dataset is split into training (80%) and testing (20%) sets. A Naive Bayes classifier is trained on the training set, and predictions are made on the test set.

### 6. Evaluation
The performance of the classifier is evaluated using:
- **Confusion Matrix**: Shows the number of true positives, true negatives, false positives, and false negatives.
- **Accuracy**: Proportion of correctly classified messages.

## Results
Example output:
```
Confusion Matrix and Statistics
          Reference
Prediction ham spam
      ham  963   15
      spam   2  134

Accuracy : 98.47%
```

Key Metrics:
- Accuracy: 98.47%
- Sensitivity: 99.79%
- Specificity: 89.93%

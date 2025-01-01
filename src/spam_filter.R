# Load required libraries
library(dplyr) # optional
library(readr)
library(caret)
library(e1071)
library(tm)

# Set the working directory to the project folder
setwd("/path/to/sms-spam-filter")

# Load data
# The dataset can be downloaded from the UCI Machine Learning Repository:
# https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
sms_data <- read_delim("data/sms_spam.txt", delim = "\t", quote = "",
                       col_names = c("Label", "Message"))

# Examine the structure of the data
str(sms_data)
table(sms_data$Label)

# Data preparation
# Convert Labels into factors
sms_data$Label <- factor(sms_data$Label)

# Define a function to clean the text data
clean_corpus <- function(corpus) {
  corpus <- tm_map(corpus, content_transformer(tolower)) # Convert to lowercase
  corpus <- tm_map(corpus, removePunctuation)            # Remove punctuation
  corpus <- tm_map(corpus, removeNumbers)                # Remove numerical digits
  corpus <- tm_map(corpus, removeWords, stopwords("en")) # Remove stop words
  corpus <- tm_map(corpus, stripWhitespace)              # Remove extra whitespace
  
  return(corpus)
}

# Create a text corpus from the messages
corpus <- VCorpus(VectorSource(sms_data$Message))
corpus_clean <- clean_corpus(corpus)

# Create a Document-Term-Matrix (DTM)
dtm <- DocumentTermMatrix(corpus_clean)

# Split the data into training and test sets (80/20)
set.seed(123)
train_indices <- createDataPartition(sms_data$Label, p = 0.8, list = FALSE)
train_data <- sms_data[train_indices, ]
test_data <- sms_data[-train_indices, ]
dtm_train <- dtm[train_indices, ]
dtm_test <- dtm[-train_indices, ]

# Select frequently used terms to reduce sparsity in the DTM
freq_terms <- findFreqTerms(dtm_train, lowfreq = 5)

# Filter DTMs to include only frequently used terms
dtm_train_freq <- dtm_train[, freq_terms]
dtm_test_freq <- dtm_test[, freq_terms]

# Convert term frequency into binary values
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  
  return(x)
}

# Apply the binary conversion to the training and test data
train_data_binary <- apply(dtm_train_freq, 2, convert_counts)
test_data_binary <- apply(dtm_test_freq, 2, convert_counts)

# Train a Naive Bayes classifier using the training data
nb_model <- naiveBayes(train_data_binary, train_data$Label)

# Make predictions on the test set
predictions <- predict(nb_model, test_data_binary)

# Evaluate the model using a confusion matrix
conf_matrix <- confusionMatrix(predictions, test_data$Label)
print(conf_matrix)

# Calculate the accuracy of the model
accuracy <- sum(predictions == test_data$Label) / nrow(test_data)
print(paste("Accuracy of the model:", round(accuracy * 100, 2), "%"))

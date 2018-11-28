################ 1. Custom Function to calculate TF, IDF and TF-IDF ################

# Function to calculate the Term Frequency by reading each row of the document.
term.frequency <- function(n) {
  n / sum(n)
}

# Function to calculate the Document Frequency.
inverse.doc.freq <- function(m) {
  corpus.size <- length(m)
  doc.count <- length(which(m > 0))
  
  log10(corpus.size / doc.count)
}

# Function to calculate the TF-IDF.
tf.idf <- function(x, idf) {
  x * idf
}

################ 2. Loading the Libraries needed ################
library(caret)
library(quanteda)
library(irlba)
library(Matrix)

################ 3. Loading the test and train datasets ################

# Read the train csv file.
text <- read.csv('data_final.csv', stringsAsFactors = FALSE)

# Add the class as a factor.
text$class <- as.factor(text$class)

# Read the test_final csv file.
test_final <- read.csv('test_final.csv', stringsAsFactors = FALSE)
test_final$class <- NA

################ 4. Splitting the train set into train and test split 80/20 ################
# Creat an index data frame to split the data into 80/20
trainIndex = createDataPartition(text$class, p=0.8, list=FALSE,times=1)

# Create new train data from the index having 80% data
new_train <- text[trainIndex,]

# Create new test data from the index having 20% data
new_test <- text[-trainIndex,]

# Removing to free the space
rm(text)

################ 5. Tokenisation and feature reduction ################

# Tokenising the text column in the final data frame for each word
# Removing numbers, removing punctuation, removing hyphensm removing symbols
train.token <- tokens(new_train$text, what="word",
                      remove_number = TRUE, remove_punct = TRUE,
                      remove_hyphens = TRUE, remove_symbols = TRUE)

# Converting all the tokens to lowercase
train.token <- tokens_tolower(train.token)

# Removing stop words from the tokens
train.token <- tokens_remove(train.token,stopwords())

# Stemming the tokens to convert into root words
train.token <- tokens_wordstem(train.token)

# Converting the tokens to document frequency matrix
mydfm <- dfm(train.token)

# Removing sparse terms by removing tokens which are less than the document frequency of 0.0005
mydfm <- dfm_trim(mydfm,min_docfreq = 0.0005, docfreq_type = "prop",verbose = TRUE)

# Calculate the term frequency of the document frequency matrix
train.tokens.df <- apply(mydfm, 1, term.frequency)

# Calculate the IDF for the tokens 
train.tokens.idf <- apply(mydfm, 2, inverse.doc.freq)

# Calculate the TF-IDF of the by passing the TF and IDF calculated earlier
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, 
                             idf = train.tokens.idf)

# The TF-IDF is transposed to represent document frequency
train.tokens.tfidf <- t(train.tokens.tfidf)

# Converting the TF-IDF matrix into Sparse matrix to reduce storage space
train.tokens.tfidf <- Matrix(train.tokens.tfidf, sparse = TRUE)

################ 6. Single Value Decomposition ################
# Performing SVD on the TF-IDF matrix by calling the irlba function
start.time <- Sys.time()
train.irlba <- irlba(t(train.tokens.tfidf), nv=300, maxit = 600)
total.time <- Sys.time() - start.time
total.time

# Saving the SVD model to load it later
saveRDS(train.irlba,"train.irlba.rds")

# Storing the v matrix of the SVD as the train
train <- data.frame(train.irlba$v)

# Attaching the class of the new_train dataframe to the train
train$y <- new_train$class

# Storing the sigma.inverse and u transpose from svd to used later for the test
sigma.inverse <- 1 / train.irlba$d
u.transpose <- t(train.irlba$u)

################ 7. Building the Classification Model ################
# Running SVM model on the train data
library(e1071)
start.time <- Sys.time()
model <- svm(y ~ . , data=train)
total.time <- Sys.time() - start.time
total.time

# Checking the summary of the model
summary(model)

# Storing the model to be used later
saveRDS(model, file = "model_final_try.rds")

################ 7. Checking the model accuracy on Test Data ################

# For the new data, we need to perform all the operations done on the train data
# 1. Tokenization
# 2. Stowords removal, lowercase, stemming on the tokens
# 3. Calculate the TF, and the TF-IDF 
# 4. Convert into SVD format
# 5. Predict labels using the model

# Tokenising the text column in the final data frame for each word
# Also removing numbers, removing punctuation, removing hyphensm removing symbols
test.tokens <- tokens(new_test$text, what = "word", 
                      remove_numbers = TRUE, remove_punct = TRUE,
                      remove_symbols = TRUE, remove_hyphens = TRUE)

# Converting all the tokens to lowercase
test.tokens <- tokens_tolower(test.tokens)

# Removing stop words from the tokens
test.tokens <- tokens_select(test.tokens, stopwords(), selection = "remove")

# Stemming the tokens to convert into root words
test.tokens <- tokens_wordstem(test.tokens, language = "english")

# Converting the tokens to document frequency matrix
testdfm <- dfm(test.tokens)

# Selecting the testdfm to have all the columns which are in the train dfm (mydfm)
testdfm <- dfm_select(testdfm, pattern = mydfm, selection = "keep")

# Calculate the IDF for the tokens 
test.tokens.df <- apply(testdfm, 1, term.frequency)

# Calculate the TF-IDF of the train tokens using the idf of the train data
test.tokens.tfidf <-  apply(test.tokens.df, 2, tf.idf, idf = train.tokens.idf)

# Transpose the tfidf
test.tokens.tfidf <- t(test.tokens.tfidf)

# Converting the tfidf matrix to a Sparse matrix to reduce storage size
test.tokens.tfidf <- Matrix(test.tokens.tfidf, sparse = TRUE)

# Converting the tfidf matrix into SVD by multiplying it with the sigma and u of the svd
test.svd.raw <- t(sigma.inverse * u.transpose %*% t(test.tokens.tfidf))

# Converting the svd matrix of the test set into a data frame
test.svd <- data.frame(as.matrix(test.svd.raw))

# Predicting the labels using the SVM model created earlier
test_pred <- predict(model, newdata = test.svd)

# Checking the confusion matrix of our model
confusionMatrix(new_test$class,test_pred)

################ 8. Predicting the Labels for the unknown data ################

# For the new data, we need to perform all the operations done on the train data
# 1. Tokenization
# 2. Stowords removal, lowercase, stemming on the tokens
# 3. Calculate the TF, and the TF-IDF 
# 4. Convert into SVD format
# 5. Predict labels using the model

# Tokenising the text column in the final data frame for each word
# Also removing numbers, removing punctuation, removing hyphensm removing symbols
test.tokens.final <- tokens(test_final$text, what = "word", 
                            remove_numbers = TRUE, remove_punct = TRUE,
                            remove_symbols = TRUE, remove_hyphens = TRUE)

# Converting all the tokens to lowercase
test.tokens.final <- tokens_tolower(test.tokens.final)

# Removing stop words from the tokens
test.tokens.final <- tokens_select(test.tokens.final, stopwords(), selection = "remove")

# Stemming the tokens to convert into root words
test.tokens.final <- tokens_wordstem(test.tokens.final, language = "english")

# Converting the tokens to document frequency matrix
testdfm.final <- dfm(test.tokens.final)

# Selecting the testdfm to have all the columns which are in the train dfm (mydfm)
testdfm.final <- dfm_select(testdfm.final, pattern = mydfm, selection = "keep")

# Calculate the IDF for the tokens 
test.tokens.df_final <- apply(testdfm.final, 1, term.frequency)

# Calculate the TF-IDF of the train tokens using the idf of the train data
test.tokens.tfidf_final <-  apply(test.tokens.df_final, 2, tf.idf, idf = train.tokens.idf)

# Transpose the tfidf
test.tokens.tfidf_final <- t(test.tokens.tfidf_final)

# Converting the tfidf matrix to a Sparse matrix to reduce storage size
test.tokens.tfidf_final <- Matrix(test.tokens.tfidf_final, sparse = TRUE)

# Converting the tfidf matrix into SVD by multiplying it with the sigma and u of the svd
test.svd.raw_final <- t(sigma.inverse * u.transpose %*% t(test.tokens.tfidf_final))

# Converting the svd matrix of the test set into a data frame
test.svd_final <- data.frame(as.matrix(test.svd.raw_final))

# Predicting the labels using the SVM model created earlier
test_pred_final <- predict(model, newdata = test.svd_final)

# Creating a data frame with the column of id and the predicted column
prediction <- data.frame(id=test_final[,1],label=test_pred_final)

# Exporting the dataframe as a txt file seperated by space
write.table(prediction,file="testing_labels_pred.txt",sep=" ",row.names = FALSE, 
                                              col.names = FALSE, quote = FALSE)

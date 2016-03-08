library(readr)
library(Rtsne)

#################################
## Run tsne on several datasets
#################################

# Load full data 
train <- read.csv("./full/train_test_full_for_tsne.csv")
# Train TSNE on full data
tsne <- Rtsne(train, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
# Save to .txt
write.csv(tsne$Y, "./full/tsne_var_full_train_test.csv")

# Load distance data 
train <- read.csv("./distance/train_test_distance_for_tsne.csv")
# Train TSNE on distance data
tsne <- Rtsne(train, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
# Save to .txt
write.csv(tsne$Y, "./distance/tsne_var_distance_train_test.csv")

# Load binary data 
train <- read.csv("./cosine/train_test_cosine_for_tsne.csv")
# Train TSNE on cosine data
dim(train)
tsne <- Rtsne(train, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
# Save to .txt
write.csv(tsne$Y, "./cosine/tsne_var_cosine_train_test.csv")

# Load binary data 
train <- read.csv("./binary/train_test_binary_for_tsne.csv")
# Train TSNE on binary data
dim(train)
tsne <- Rtsne(train, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
# Save to .txt
write.csv(tsne$Y, "./binary/tsne_var_binary_train_test.csv")

# Load ternary data 
train <- read.csv("./ternary/train_test_ternary_for_tsne.csv")
# Train TSNE on ternary data
dim(train)
tsne <- Rtsne(train, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
# Save to .txt
write.csv(tsne$Y, "./ternary/tsne_var_ternary_train_test.csv")
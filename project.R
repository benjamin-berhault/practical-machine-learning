rm(list=ls(all=TRUE))

## Install the package if needed
pkgTest <- function(x)
{
  if (!require(x,character.only = TRUE))
  {
    install.packages(x,dep=TRUE)
    if(!require(x,character.only = TRUE)) stop("Package not found")
  }
}

# load the required packages
pkgTest("caret")
pkgTest("rattle")
pkgTest("rpart")
pkgTest("rpart.plot")
pkgTest("RColorBrewer")
pkgTest("randomForest")
pkgTest("gbm");

### Getting and loading the data
trainURL <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainingFile <- "./pml-training.csv"
testingFile  <- "./pml-testing.csv"
if (!file.exists(trainingFile)) {download.file(trainURL, destfile=trainingFile)}
if (!file.exists(testingFile)) {download.file(testURL, destfile=testingFile)}


training <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
testing <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))

### Cleaning the data

# Remove NearZeroVariance variables
# nearZeroVar() : By default, a predictor is classified as near-zero variance 
# if the percentage of unique values in the samples is less than 10% 
# and when the frequency ratio mentioned above is greater than 19 95/5.

# nzv : collect column near zero variance info
nzv <- nearZeroVar(training, saveMetrics=TRUE)
# save those column names
non_relevant_columns <- c(names(training[,nzv$nzv==TRUE]))
# delete those columns
training <- training[,nzv$nzv==FALSE]

non_relevant_columns <- c(non_relevant_columns,names(training[1]))
training <- training[c(-1)]

## We investigate missing values
# Quantity of NAs by column
training_nb_of_NAs <- apply(training, 2, function(x) sum(is.na(x)))

# We check the different amount of missing observations
training_dif_amount_of_NAs <- unique(training_nb_of_NAs)
pourcent_miss_obs <- round(training_dif_amount_of_NAs/nrow(training)*100,2)
pourcent_miss_obs[order(pourcent_miss_obs)]

# Smallest amount of missing observations 
pourcent_miss_obs[order(pourcent_miss_obs)][2]


# The pourcentage is so important that we remove each column with missing values.
columns_2_ignore <- c()
# columns to ignore
column_with_NA <- (training_nb_of_NAs != 0)
for (i in 1:length(column_with_NA)) {
  if (column_with_NA[i] == TRUE){
    # store the name of column to remove
    columns_2_ignore <- c(columns_2_ignore, names(column_with_NA[i]))
  }
}

# store the name of irrelevant columns
non_relevant_columns <- c(non_relevant_columns,columns_2_ignore)

# remove irrelevant columns from the training dataset
training <- training[, !(colnames(training) %in% columns_2_ignore), drop=FALSE]


## 3. Split the dataset
# Split the intitial training dataset into two parts : pre-training set and validation set.
# We use the function "createDataPartition()" (from the Caret package) to have balanced splits of the data.
set.seed(1982)
inTrain <- createDataPartition(training$classe, p=0.6, list=FALSE)
pre_training <- training[inTrain, ]
validation_set <- training[-inTrain, ]

# pre-training dataset's dimensions 
dim(pre_training)
# validation dataset's dimensions 
dim(validation_set)


# Reduce the testing dataset to relevant columns
clean2 <- colnames(pre_training[, -58])  # remove the classe column
testing <- testing[clean2]

# testing dataset's dimensions
dim(testing)


## 4. Build the 3 different predictive models
### Decision Tree
set.seed(1982)
# xval : define the number of cross-validations
fitControl1 = rpart.control(cp = 0, xval = 5)
mod_decisionTree <- rpart(classe ~ ., data=pre_training, method="class", control = fitControl1)
fancyRpartPlot(mod_decisionTree)

prediction_DT <- predict(mod_decisionTree, validation_set, type = "class")
cmtree <- confusionMatrix(prediction_DT, validation_set$classe)
cmtree
par(mfrow=c(1,1))
plot(cmtree$table, col = cmtree$byClass, main = paste("Decision Tree Confusion Matrix"))

# Decision Tree accuracy
round(100*cmtree$overall['Accuracy'],2)

### Random Forests

set.seed(1982)

mod_randomForest <- randomForest(classe ~ ., data=pre_training)
prediction_RF <- predict(mod_randomForest, validation_set, type = "class")
cmrf <- confusionMatrix(prediction_RF, validation_set$classe)
cmrf

par(mfrow=c(1,1))
plot(mod_randomForest)
plot(cmrf$table, col = cmtree$byClass, main = paste("Random Forest Confusion Matrix: Accuracy =", round(cmrf$overall['Accuracy'], 4)))

# Random Forests accuracy
round(100*cmrf$overall['Accuracy'],2)

### Generalized Boosted Regression
set.seed(1982)
# method : cross validation resampling method | number : number of resampling iterations
fitControl2 <- trainControl(method = "cv",
                            number = 5)

# method : Generalized Boosted Regression Models
mod_gradientBoostR <- train(classe ~ ., data=pre_training, method = "gbm",
                            trControl = fitControl2,
                            verbose = FALSE)

prediction_GBR <- predict(mod_gradientBoostR, newdata=validation_set)
gbmAccuracyTest <- confusionMatrix(prediction_GBR, validation_set$classe)
gbmAccuracyTest

plot(mod_gradientBoostR, ylim=c(0.9, 1))

# Generalized Boosted Regression accuracy
round(100*gbmAccuracyTest$overall['Accuracy'],2)

# Predicting results on test data
# test set doesn't have some of the levels present in training. 
# So to solve this we use :
for (i in 1:(length(testing)-1)) {
  levels(testing[[i]]) <- levels(pre_training[[i]])
}

prediction_RF_submit <- predict(mod_randomForest, testing[-c(58)])
testing$classe <- prediction_RF_submit

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(prediction_RF_submit)







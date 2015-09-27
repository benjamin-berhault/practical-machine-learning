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

set.seed(1982)

### Getting and loading the data

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

# Remove the ID column (non relevant)
non_relevant_columns <- c(non_relevant_columns,names(training[1]))
training <- training[c(-1)]
non_relevant_columns
dim(training)

# Quantity of NAs by column
training_nb_of_NAs <- apply(training, 2, function(x) sum(is.na(x)))

# We check the different amount of missing observations
training_dif_amount_of_NAs <- unique(training_nb_of_NAs)

# Smallest amount of missing observations 
smallest_missing <- min(training_dif_amount_of_NAs)

# Pourcentage of missing observations for that column 
round(smallest_missing/nrow(training)*100,2)

# The pourcentage is so important that we remove 
# each column with missing values

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

# remove irrelevant columns from the dataset
training <- training[, !(colnames(training) %in% columns_2_ignore), drop=FALSE]


### Partioning the training set in one pre-training set and one validation set
set.seed(1982)
inTrain <- createDataPartition(training$classe, p=0.6, list=FALSE)
pre_training <- training[inTrain, ]
validation_set <- training[-inTrain, ]
dim(pre_training)
dim(validation_set)


# reduce the testing dataset to relevant columns
clean2 <- colnames(pre_training[, -58])  # remove the classe column
testing <- testing[clean2]
dim(testing)

### Using ML algorithms for prediction: Decision Tree
modFitA1 <- rpart(classe ~ ., data=pre_training, method="class")
fancyRpartPlot(modFitA1)

prediction_DT <- predict(modFitA1, validation_set, type = "class")
cmtree <- confusionMatrix(prediction_DT, validation_set$classe)
cmtree
par(mfrow=c(1,1))
plot(cmtree$table, col = cmtree$byClass, main = paste("Decision Tree Confusion Matrix"))

### Using ML algorithms for prediction: Random Forests
set.seed(1982)
mod_randomForest <- randomForest(classe ~ ., data=pre_training)
prediction_RF <- predict(mod_randomForest, validation_set, type = "class")
cmrf <- confusionMatrix(prediction_RF, validation_set$classe)
cmrf

par(mfrow=c(1,1))
plot(mod_randomForest)
plot(cmrf$table, col = cmtree$byClass, main = paste("Random Forest Confusion Matrix: Accuracy =", round(cmrf$overall['Accuracy'], 4)))


### Using gbm prediction
set.seed(1982)
# method : cross validation resampling method | number : number of resampling iterations
fitControl <- trainControl(method = "cv",
                           number = 5)

# method : Generalized Boosted Regression Models
mod_gradientBoostR <- train(classe ~ ., data=pre_training, method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE)

prediction_GBR <- predict(mod_gradientBoostR, newdata=validation_set)
gbmAccuracyTest <- confusionMatrix(prediction_GBR, validation_set$classe)
gbmAccuracyTest

plot(mod_gradientBoostR, ylim=c(0.9, 1))

### Predicting results on test data
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
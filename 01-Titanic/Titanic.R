library(tidyverse)
library(tidyr)
library(gridExtra)
library(caret)
library(randomForest)
library(car)
library(xgboost)

# load train data
## set the file path
file = "https://raw.githubusercontent.com/Carloszone/Kaggle-Cases/main/01-Titanic/train.csv"


## load data and name it "dat_train"
dat_train = read.csv(file)


## repeat for test set
file = "https://raw.githubusercontent.com/Carloszone/Kaggle-Cases/main/01-Titanic/test.csv"

dat_test = read.csv(file)



# Exploratory Data Analysis
str(dat_train)

## basic descriptive statistics for all variables
psych::describe(dat_train)


## distribution plot for variables
factor_names <- c("Survived", "Pclass", "Sex", "SibSp", "Parch", "Embarked")
numeric_names <- c("Age", "Fare")

auto_histogram <- function(names){
  for(name in names){
    name <- sym(name)
    p <- dat_train %>%
      ggplot(aes_string(x = name)) +
      geom_histogram(stat = "count", fill = "blue", alpha = 0.7)
    show(p)
  }
}

auto_boxplot <- function(names){
  for(name in names){
    name <- sym(name)
    p <- dat_train %>% select(name) %>% 
      gather(key = "variables", value = "value") %>%
      ggplot(aes(x = variables, y = value)) +
      geom_boxplot()
    show(p)
  }
}

auto_histogram(factor_names)
auto_boxplot(numeric_names)


## plots between survived and other variables
### create boxplot to check the relationship between survived and numeric variables
p1 <- dat_train %>%
  ggplot(aes(x = factor(Survived), y = Age)) +
  geom_boxplot()
p2 <- dat_train %>%
  ggplot(aes(x = factor(Survived), y = Fare)) +
  geom_boxplot()

grid.arrange(p1, p2, nrow = 1)


### create bar plot to check the relationship between survived and factor variables

p3 <- dat_train %>%
  ggplot(aes(x = Pclass, fill = factor(Survived))) +
  geom_bar()

p4 <- dat_train %>%
  ggplot(aes(x = Sex, fill = factor(Survived))) +
  geom_bar()

p5 <- dat_train %>%
  ggplot(aes(x = SibSp, fill = factor(Survived))) +
  geom_bar()

p6 <- dat_train %>%
  ggplot(aes(x = Parch, fill = factor(Survived))) +
  geom_bar()

p7 <- dat_train %>%
  ggplot(aes(x = Embarked, fill = factor(Survived))) +
  geom_bar()

grid.arrange(p3, p4, p5, p6, p7, nrow = 2)


# Data Pre-processing
## deal with missing values
dat_train$Age[is.na(dat_train$Age)] <- median(dat_train$Age, na.rm = TRUE)
is.na(dat_train$Age) %>% sum()

dat_train$Embarked[dat_train$Embarked == ""] <- "S"

## deal with outliers
### do nothing first


## deal with categorical variables
### recode Sex column
dat_train$Sex <- ifelse(dat_train$Sex == "female", 0, 1)

### generate dummy variables
auto_dummy <- function(data, colnamelist){
  new <- data
  for(name in colnamelist){
    for(value in unique(new[,name])){
      cname <- paste(name, value, sep = "")
      new[,cname] <- ifelse(new[,name] == value, 1, 0)
    }
  }
  return(new)
}

dummylist <- c("Pclass", "SibSp", "Parch", "Embarked")


dat_new <- auto_dummy(dat_train, dummylist)
dat_new <- dat_new[,-c(1, 3, 4, 7, 8, 9, 11, 12)]
# Feature Engineering
## feature selection
model_rf <- randomForest(Survived~., data = dat_new)

model_rf$importance

## collinear check
dat_new[,-c(1,2)] %>% cor(., method = "pearson") %>% 
  pheatmap::pheatmap(., display_numbers = T)

## based on the importance and heatmap, encode variables
dat_new <- dat_train
dat_new$SibSp <- ifelse(dat_new$SibSp == 0, 1, 0)
dat_new$Parch <- ifelse(dat_new$Parch == 0, 1, 0)
dat_new$Embarked <- ifelse(dat_new$Embarked == "Q", 1, 0)

## repeat
dat_new <- dat_new[,-c(1, 4, 9, 11)]

dat_new$Survived <- as.factor(dat_new$Survived)

model_rf <- randomForest(Survived~., data = dat_new)

model_rf$importance

## collinear check
dat_new[,-1] %>% cor(., method = "pearson") %>% 
  pheatmap::pheatmap(., display_numbers = T)



# Model Selection
## first use 10 common model to train
model_names <- c("glm", "lda", "naive_bayes", "svmLinear", "knn", "gamLoess", "multinom", "qda", "rpart", "adaboost")

## start to train
train_control <- trainControl(method="cv", number=5)

fits <- lapply(model_names, function(model){ 
  print(model)
  train(Survived ~ ., method = model, trControl=train_control, data = dat_new)
}) 

model_accuraries <- sapply(fits, function(model){
  res <- (predict(model, dat_new) == dat_new$Survived) %>% mean()
  return(res)
})

## store and show the result
model_result <- data.frame(models = model_names, accuracies = model_accuraries)

model_result %>% arrange(desc(accuracies))


## use Gradient Boosting to train
### split data into train set and test set
set.seed(1, sample.kind = "Rounding")
index <- sample((1:nrow(dat_new)), round(0.8*nrow(dat_new)))
training <- dat_new[index,]
testing <- dat_new[-index,]

training_x <- training[,-1] %>% as.matrix()
training_y <- training[,1]
testing_x <- testing[,-1] %>% as.matrix()
testing_y <- testing[, 1]
### sample case of xgboost
param <- list(eta = 1,
              subsample = 1,
              colsample_bytree = 1,
              max_depth = 6)
model_gb <- xgboost(data = training_x, label =training_y, params = param, nrounds = 500)
predictions <- predict(model_gb, testing_x)
predictions <- ifelse(predictions > 0.5, 1, 0)
accuracy <- (predictions == testing_y) %>% mean()


### Grid Search
#### determine nrounds
param <- list(eta = 1,
              subsample = 1,
              colsample_bytree = 1,
              max_depth = 6)
n <- seq(1,100, 5)
accuracy <- sapply(n, function(npar){
  model_gb <- xgboost(data = training_x, label =training_y, params = param, nrounds = npar)
  predictions <- predict(model_gb, testing_x)
  predictions <- ifelse(predictions > 0.5, 1, 0)
  accuracy <- (predictions == testing_y) %>% mean()
  return(accuracy)
})
plot(n, accuracy, main = "find best nrouds") # the best nround is 1


#### determine max_depth
n = seq(1,20,1)
accuracy <- sapply(n, function(npar){
  param <- list(eta = 0.1,
                subsample = 1,
                colsample_bytree = 1,
                max_depth = npar)
  model_gb <- xgboost(data = training_x, label =training_y, params = param, nrounds = 1)
  predictions <- predict(model_gb, testing_x)
  predictions <- ifelse(predictions > 0.5, 1, 0)
  accuracy <- (predictions == testing_y) %>% mean()
  return(accuracy)
})

plot(n, accuracy, main = "find best max_depth") # the best max_depth is 6


#### determine colsample_bytree
n = seq(0,1,0.1)
accuracy <- sapply(n, function(npar){
  param <- list(eta = 0.1,
                subsample = 1,
                colsample_bytree = npar,
                max_depth = 6)
  model_gb <- xgboost(data = training_x, label =training_y, params = param, nrounds = 1)
  predictions <- predict(model_gb, testing_x)
  predictions <- ifelse(predictions > 0.5, 1, 0)
  accuracy <- (predictions == testing_y) %>% mean()
  return(accuracy)
})

plot(n, accuracy, main = "find best colsample_bytree") # the best colsample_bytree is 1


#### determine subsample
n = seq(0,1,0.01)
accuracy <- sapply(n, function(npar){
  param <- list(eta = 0.1,
                subsample = npar,
                colsample_bytree = 1,
                max_depth = 6)
  model_gb <- xgboost(data = training_x, label =training_y, params = param, nrounds = 1)
  predictions <- predict(model_gb, testing_x)
  predictions <- ifelse(predictions > 0.5, 1, 0)
  accuracy <- (predictions == testing_y) %>% mean()
  return(accuracy)
})

data.frame(n = n, accuracy = accuracy) %>% arrange(desc(accuracy)) # the best subsample is 0.92


#### determine eta
n = seq(0,1,0.001)
accuracy <- sapply(n, function(npar){
  param <- list(eta =npar,
                subsample = 0.92,
                colsample_bytree = 1,
                max_depth = 6)
  model_gb <- xgboost(data = training_x, label =training_y, params = param, nrounds = 1)
  predictions <- predict(model_gb, testing_x)
  predictions <- ifelse(predictions > 0.5, 1, 0)
  accuracy <- (predictions == testing_y) %>% mean()
  return(accuracy)
})

data.frame(n = n, accuracy = accuracy) %>% 
  arrange(desc(accuracy)) %>% top_n(10) # the best eta is 0.655


### create the final xgboost model
param <- list(eta =0.655,
              subsample = 0.92,
              colsample_bytree = 1,
              max_depth = 6)
model_gb <- xgboost(data = training_x, label =training_y, params = param, nrounds = 1)
predictions <- predict(model_gb, testing_x)
predictions <- ifelse(predictions > 0.5, 1, 0)
accuracy <- (predictions == testing_y) %>% mean()

## use random tree to train
### Grid Search
#### determine ntree
grid <- expand.grid(.mtry=sqrt(ncol(training_x)))
ntree <- c(100,1000,2000,5000)
accuracy <- sapply(ntree, function(ntr){
  model_rf <- train(training_x, factor(training_y), 
                    method = "rf", ntree = ntr, 
                    tuneGrid = grid, trControl = train_control)
  accuracy <- (predictions == testing_y) %>% mean()
  return(accuracy)
})
plot(ntree, accuracy)
data.frame(ntree = ntree, accuracy = accuracy) %>% 
  arrange(desc(accuracy)) %>% top_n(10) # each ntree get the same result



# Ensemble Generation
ensemble <- function(train_x, train_y, test_x){
  # random forest
  grid <- expand.grid(.mtry=sqrt(ncol(training_x)))
  model_rf <- train(x = train_x, y = train_y, method = "rf", 
                    tuneGrid = grid, trControl = train_control)
  pred_rf <- predict(model_rf, test_x) == 1
  
  # graident boosting
  param <- list(eta =0.655,
                subsample = 0.92,
                colsample_bytree = 1,
                max_depth = 6)
  model_gb <- xgboost(data = train_x, label = train_y, params = param, nrounds = 1)
  pred_gb <- ifelse(predict(model_gb, test_x) > 0.5, 1, 0) == 1
    
  # 10 common model
  model_names <- c("gamLoess", "qda", "rpart", "adaboost")

  train_control <- trainControl(method="cv", number=5)
  
  fits <- lapply(model_names, function(model){ 
    print(model)
    train(x = train_x, y = train_y, method = model, trControl=train_control)
  }) 
  # store all predictions
  
  pred_all <- data.frame(rf = pred_rf, gb = pred_gb)
  for(model in fits){
    predictions <- predict(model, test_x)
    pred_all <- cbind(pred_all, pred = predictions == 1)
  }
  
  final_pred <- ifelse(rowMeans(pred_all) >0.5, 1, 0)
  return(final_pred)
}

final <- ensemble(training_x, factor(training_y), training_x)


# calculate the result
## deal with the test set
dat_test$Sex <- ifelse(dat_test$Sex == "female", 0, 1)
dat_test$SibSp <- ifelse(dat_test$SibSp == 0, 1, 0)
dat_test$Parch <- ifelse(dat_test$Parch == 0, 1, 0)
dat_test$Embarked <- ifelse(dat_test$Embarked == "Q", 1, 0)

dat_test$Age[is.na(dat_test$Age)] <- median(dat_test$Age, na.rm = TRUE)
is.na(dat_test$Age) %>% sum()

dat_test$Fare[is.na(dat_test$Fare)] <- median(dat_test$Fare, na.rm = TRUE)
is.na(dat_test$Fare) %>% sum()

dat_test$Embarked[dat_test$Embarked == ""] <- "S"


dat_test$Pclass <- as.numeric(dat_test$Pclass)
dat_test$Embarked <- as.numeric(dat_test$Embarked)
test <- dat_test[,-c(1,3,8,10)] %>% as.matrix()

train_x <- dat_new[,-1] %>% as.matrix()
train_y <- dat_new[,1]
res <- ensemble(train_x, factor(train_y), test)

result <- data.frame(PassengerId = dat_test$PassengerId, Survived = res)
write.csv(result, "result_02.csv", row.names = FALSE)

library(tidyverse)
library(tidyr)
library(gridExtra)
library(caret)
library(randomForest)
library(xgboost)
library(extraTrees)

# load train data
## set the file path
file = "https://raw.githubusercontent.com/Carloszone/Kaggle-Cases/main/01-Titanic/train.csv"


## load data and name it "dat_train"
dat_train = read.csv(file)


## repeat for test set
file = "https://raw.githubusercontent.com/Carloszone/Kaggle-Cases/main/01-Titanic/test.csv"

dat_test = read.csv(file)

## join two sets
Survived <- dat_train$Survived

dat <- rbind(dat_train[,-2], dat_test)
dat %>% head()
## delete ID and string columns because I don't know how to use them
deletelist <- c("PassengerId", "Name", "Ticket", "Cabin")
dat <- dat[,!(names(dat) %in% deletelist)]





# Exploratory Data Analysis
dat %>% head()
str(dat)

## check the NA and missing value
sapply(dat, function(col){
  is.na(col) %>% sum()
})

sapply(dat, function(col){
  (col[!is.na(col)] == "") %>% sum()
})

## distribution plot for variables
table(Survived) # most people died in the crash

factor_names <- c("Pclass", "Sex", "SibSp", "Parch", "Embarked")
numeric_names <- c("Age", "Fare")

auto_histogram <- function(names){
  for(name in names){
    name <- sym(name)
    p <- dat %>%
      ggplot(aes_string(x = name)) +
      geom_histogram(stat = "count", fill = "blue", alpha = 0.7)
    show(p)
  }
}

auto_boxplot <- function(names){
  for(name in names){
    name <- sym(name)
    p <- dat %>% select(name) %>% 
      gather(key = "variables", value = "value") %>%
      ggplot(aes(x = variables, y = value)) +
      geom_boxplot()
    show(p)
  }
}

auto_histogram(factor_names)
auto_boxplot(numeric_names)

## finding relationship between survived and other variables
train <- cbind(Survived, dat[1:length(Survived),])

## Pclass and Survived
train %>%
  ggplot(aes(x = Pclass, fill = factor(Survived))) +
  geom_bar() # with the Pclass number add, the death rate increasing / rich people have more chance to survive

## Sex and Survived
train %>% ggplot(aes(x = Sex, fill = factor(Survived))) +
  geom_bar() # female group has higher rate to survive

## Age and Survived
train %>% ggplot(aes(x = Age, fill = factor(Survived))) +
  geom_bar(stat = "count", width = 5, alpha = 0.3) # baby and elders have more opportunities to survive

## Sibsp and Survived
train %>% select(SibSp, Survived) %>%
  group_by(SibSp) %>%
  summarise(mean(Survived), n = n()) # having siblings can improve the survived rate

## Parch and Survived
train %>% select(Parch, Survived) %>%
  group_by(Parch) %>%
  summarise(n = n(), mean(Survived)) # traveling with parents also can improve the survived rate

## Fare and Survived
train %>% ggplot(aes(x = Fare, color = factor(Survived))) +
  geom_line(stat = "count") # like pclass, rich people who pay more money for fare can survive with a higher rate

## Embarked and Survived
train %>% select(Embarked, Survived) %>%
  group_by(Embarked) %>%
  summarise(n = n(), mean(Survived)) # ignoring the empty value, if a people embarked in the C, he will have a high rate to survive. But why?



# Data Pre-processing
## encode
dat$Sex <- ifelse(dat$Sex == "female", 0, 1) # 0: female; 1: male
dat$Embarked[dat$Embarked == ""] <- "Z" # set the ""value to "Zero"

## deal with missing values
### Age with knn
fit <- dat %>% 
  select(Pclass, Sex, Age, SibSp, Parch, Fare) %>%
  filter(!is.na(Age) & !is.na(Fare)) %>%
  train(Age ~ ., data = ., method = "knn")

dat$Age[is.na(dat$Age)] <- dat %>% 
  select(Pclass, Sex, Age, SibSp, Parch, Fare) %>%
  filter(is.na(Age)) %>%
  predict(fit, .)

dat$Age %>% is.na() %>% sum()

### Fare. there is only one missing value, so I use the median Fare
dat$Fare[is.na(dat$Fare)] <- median(dat$Fare, na.rm = TRUE)
dat$Fare %>% is.na() %>% sum()

## generate dummy variables
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

colnames <- c("SibSp", "Parch", "Embarked")
dat <- auto_dummy(dat,colnames)


## create correlation coefficient matrices
dat[,-c(1,4,5,7)] %>% cor(., method = "pearson") %>% 
  pheatmap::pheatmap(., display_numbers = T)

### add survived and only use the train set data
train <- cbind(Survived, dat[1:length(Survived),])[,-c(2,5,6,8,23)]
train %>% cor(., method = "pearson") %>% 
  pheatmap::pheatmap(., display_numbers = T)





# Feature Engineering
train_x <- dat[,-c(4,5,7)][1:length(Survived),]
train_y <- train[,1] %>% factor()
test_x <- dat[,-c(4,5,7)][-(1:length(Survived)),]

## find feature by randomForest
model_rf <- randomForest(train_x, train_y)
model_rf$importance %>% as.data.frame() %>% arrange(desc(MeanDecreaseGini))




# Model Selection
## CV setting
train_control <- trainControl(method="cv", number=5)

## Grid Search
### rf
#### mtry
grid <- expand.grid(mtry= 1:23)
model_rf <- train(train_x, 
                  train_y,
                  method = "rf",
                  tuneGrid = grid,
                  trControl = train_control)
model_rf$bestTune # best mtry is 5


#### ntree
ntrees <- seq(100,1000,100)
grid <- expand.grid(mtry= 5)
res <- sapply(ntrees, function(ntree){
  model_rf <- train(train_x, 
                  train_y,
                  method = "rf",
                  tuneGrid = grid,
                  ntree = ntree,
                  trControl = train_control)
  return(model_rf$results$Accuracy)
})
plot(ntrees, res) # best ntree is 900

#### final rf model
grid <- expand.grid(mtry= 5)
model_rf <- train(train_x, 
                  train_y,
                  method = "rf",
                  tuneGrid = grid,
                  ntree = 900,
                  trControl = train_control)
model_rf$results["Accuracy"]
predict(model_rf, test_x) 



### rpart
#### cp
grid <- expand.grid(cp = seq(0,0.5, 0.005))
model_rpart <- train(train_x, 
                  train_y,
                  method = "rpart",
                  tuneGrid = grid,
                  trControl = train_control)
model_rpart$bestTune # the best cp is 0.005

#### final model
grid <- expand.grid(cp = 0.005)
model_rpart <- train(train_x, 
                     train_y,
                     method = "rpart",
                     tuneGrid = grid,
                     trControl = train_control)
model_rpart$results["Accuracy"]
predict(model_rpart, test_x) 

### adaboost
grid <- expand.grid(nIter = floor((1:10) * 25), method = c("Adaboost.M1", "Real adaboost"))
model_ada <- train(train_x, 
                   train_y,
                   method = "adaboost",
                   tuneGrid = grid,
                   trControl = train_control)
model_ada$bestTune # the best nIter is 25 and the method is "Adaboost.M1"

#### final model
grid <- expand.grid(nIter = 25, method = "Adaboost.M1")
model_ada <- train(train_x, 
                   train_y,
                   method = "adaboost",
                   tuneGrid = grid,
                   trControl = train_control)
model_ada$results["Accuracy"]
predict(model_ada, test_x) 


### xgbTree
grid <- expand.grid(nrounds = seq(10,200,20),
                    max_depth = 1:5,
                    eta = 0.4,
                    gamma = 0,
                    colsample_bytree = seq(0.5,1, 0.05),
                    min_child_weight = 1,
                    subsample = seq(0.5,1, 0.05))
model_xgb <- train(train_x, 
                   train_y,
                   method = "xgbTree",
                   tuneGrid = grid,
                   trControl = train_control)
model_xgb$bestTune # best nround = 30, max_depth = 5, gamma = 0, colsample_bytree = 0.55, min = 1, subsample = 0.7


grid <- expand.grid(nrounds = 30,
                    max_depth = 5,
                    eta = seq(0.01,0.4,0.01),
                    gamma = 0,
                    colsample_bytree = 0.55,
                    min_child_weight = 1,
                    subsample = 0.7)
model_xgb <- train(train_x, 
                   train_y,
                   method = "xgbTree",
                   tuneGrid = grid,
                   trControl = train_control)
model_xgb$bestTune # the best eta is 0.13

#### final model
grid <- expand.grid(nrounds = 30,
                    max_depth = 5,
                    eta = 0.13,
                    gamma = 0,
                    colsample_bytree = 0.55,
                    min_child_weight = 1,
                    subsample = 0.7)
model_xgb <- train(train_x, 
                   train_y,
                   method = "xgbTree",
                   tuneGrid = grid,
                   trControl = train_control)
model_xgb$results["Accuracy"]
predict(model_xgb, test_x) 




# Ensemble
ensemble <- function(train_x, train_y, test_x){
  ## Random Forest
  grid <- expand.grid(mtry= 5)
  model_rf <- train(train_x, 
                    train_y,
                    method = "rf",
                    tuneGrid = grid,
                    ntree = 900,
                    trControl = train_control)
  pred_rf <- predict(model_rf, test_x) == 1

  
  ## rpart
  grid <- expand.grid(cp = 0.005)
  model_rpart <- train(train_x, 
                       train_y,
                       method = "rpart",
                       tuneGrid = grid,
                       trControl = train_control)
  pred_rpar <- predict(model_rpart, test_x) == 1
  
  ## xgboost
  grid <- expand.grid(nrounds = 30,
                      max_depth = 5,
                      eta = 0.13,
                      gamma = 0,
                      colsample_bytree = 0.55,
                      min_child_weight = 1,
                      subsample = 0.7)
  model_xgb <- train(train_x, 
                     train_y,
                     method = "xgbTree",
                     tuneGrid = grid,
                     trControl = train_control)
  pred_xgb <- predict(model_xgb, test_x) == 1
  
  pred_all <- cbind(pred_rf, pred_rpar, pred_xgb)
  
  res <- ifelse(rowSums(pred_all) > 1.5, 1, 0)
  
  return(res)
}

res <- ensemble(train_x, train_y, test_x)
res <- data.frame(PassengerId = dat_test$PassengerId, Survived = res)
write.csv(res, "pred_10.csv", row.names = FALSE)
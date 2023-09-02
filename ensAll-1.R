# Load the MNIST digit recognition dataset into R
# http://yann.lecun.com/exdb/mnist/
# assume you have all 4 files and gunzip'd them
# creates train$n, train$x, train$y  and test$n, test$x, test$y
# e.g. train$x is a 60000 x 784 matrix, each row is one digit (28x28)
# call:  show_digit(train$x[5,])   to see a digit.
# brendan o'connor - gist.github.com/39760 - anyall.org

load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file('mnist1/train-images-idx3-ubyte')
  test <<- load_image_file('mnist1/t10k-images-idx3-ubyte')
  
  train$y <<- load_label_file('mnist1/train-labels-idx1-ubyte')
  test$y <<- load_label_file('mnist1/t10k-labels-idx1-ubyte')  
}


show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

train <- data.frame()
test <- data.frame()

# Load data.
load_mnist()

########################### Bagging ############################
#####Naive Bayes Classifier
library(e1071)
library(foreach)
inTrain = data.frame(y=train$y[train$y  %in% c(2,3)], train$x[train$y  %in% c(2,3),])
inTest = data.frame(y=test$y[test$y  %in% c(2,3)], test$x[test$y  %in% c(2,3),])
inTrain$y <- factor(inTrain$y)
length_divisor <- 100
bts <- 50
misClass <- vector('numeric', length=bts)
##a
predictions<-foreach(m=1:bts,.combine=cbind) %do%  {
  sampleRows <- sample(nrow(inTrain), size=floor(nrow(inTrain)/length_divisor))
  fit <- naiveBayes(y ~ ., method='class', data = inTrain[sampleRows,])
  predictions <- predict(fit, newdata = inTest, type = "class")
}
##b
for (i in 1:bts)  {
  if (i==1){
    tab <- table(predictions[,1],inTest$y)
  } else {
    prmv <- apply(predictions[,1:i],1,function(x) names(which.max(table(x))))
    tab <- table(prmv,inTest$y)
  }
  misClass[i] = 1-sum(diag(tab))/sum(tab)
}
plot(1:bts, misClass,main = "Bagging", xlab = "Number of Datasets", ylab = "Misclassification Rate", type="l",col = "red")

#####Tree Classifier
library(foreach)
library(rpart)
# Setup training data with digit and pixel values with 60/40 split for train/cv.
inTrain = data.frame(y=train$y[train$y  %in% c(2,3)], train$x[train$y  %in% c(2,3),])
inTest = data.frame(y=test$y[test$y  %in% c(2,3)], test$x[test$y  %in% c(2,3),])
length_divisor <- 100
bts <-50
misClass1 <- vector('numeric', length=bts)
##a
predictions<-foreach(m=1:bts,.combine=cbind) %do%  {
  # using sample function without seed
  sampleRows <- sample(nrow(inTrain), size=floor(nrow(inTrain)/length_divisor))
  model <- rpart(y ~ ., method='class', data = inTrain[sampleRows,])
  predictions <- predict(model,inTest, type = "class")
}
#b
for (i in 1:bts)  {
  if (i==1){
    tab <- table(predictions[,1],inTest$y)
  } else {
    prmv <- apply(predictions[,1:i],1,function(x) names(which.max(table(x))))
    tab <- table(prmv,inTest$y)
  }
  misClass1[i] = 1-sum(diag(tab))/sum(tab)
}
par(new = TRUE)
plot(1:bts, misClass1,main = "Bagging", xlab = "Number of Datasets", ylab = "Misclassification Rate", type="l",col = "green")


########################### Random Forest ############################
##a
library(foreach)
library(rpart)
inTrain = data.frame(y=train$y[train$y  %in% c(2,3)], train$x[train$y  %in% c(2,3),])
inTest = data.frame(y=test$y[test$y  %in% c(2,3)], test$x[test$y  %in% c(2,3),])
length_divisor <- 100
bts <-50
misClass2 <- vector('numeric', length=bts)

predictions<-foreach(m=1:bts,.combine=cbind) %do%  {
  sampleCols <- sample(2:ncol(inTrain), 50)
  sampleRows <- sample(nrow(inTrain), size=floor(nrow(inTrain)/length_divisor))
  model <- rpart(y ~ ., method='class', data = inTrain[sampleRows,c(1,sampleCols)])
  predictions <- predict(model,inTest, type = "class")
}

for (i in 1:bts)  {
  if (i==1){
    tab <- table(predictions[,1],inTest$y)
  } else {
    prmv <- apply(predictions[,1:i],1,function(x) names(which.max(table(x))))
    tab <- table(prmv,inTest$y)
  }
  misClass2[i] = 1-sum(diag(tab))/sum(tab)
}

##b
library(foreach)
library(rpart)
library(pROC)
library(caret)
inTrain = data.frame(y=train$y[train$y  %in% c(2,3)], train$x[train$y  %in% c(2,3),])
inTest = data.frame(y=test$y[test$y  %in% c(2,3)], test$x[test$y  %in% c(2,3),])
length_divisor <- 100
bts <-50
fn <- c(10,50,300)
misClass2 <- matrix(, nrow = bts, ncol = length(fn))
j <- 1
for (f in fn) {
  predictions<-foreach(m=1:bts,.combine=cbind) %do%  {
    sampleCols <- sample(2:ncol(inTrain), f)
    sampleRows <- sample(nrow(inTrain), size=floor(nrow(inTrain)/length_divisor))
    model <- rpart(y ~ ., method='class', data = inTrain[sampleRows,c(1,sampleCols)])
    predictions <- predict(model,inTest, type = "class")
  }
  
  for (i in 1:bts)  {
    if (i==1){
      tab <- table(predictions[,1],inTest$y)
    } else {
      prmv <- apply(predictions[,1:i],1,function(x) names(which.max(table(x))))
      tab <- table(prmv,inTest$y)
    }
    misClass2[i,j] = 1-sum(diag(tab))/sum(tab)
  }
  j <- j+1
}

library(ggplot2)
library(reshape2)
x<-1:50
df <- data.frame(x,misClass,misClass1,misClass2[,1],misClass2[,2],misClass2[,3])
names(df) <- c("x","NB Bag","Tree Bag","Tree Bag 10","Tree Bag 50", "Tree Bag 300")
df2 <- melt(data = df, id.vars = "x")
ggplot(data = df2, aes(x = x, y = value, colour = variable)) + geom_line()

##c
library(randomForest)
inTrain = data.frame(y=train$y[train$y  %in% c(2,3)], train$x[train$y  %in% c(2,3),])
inTest = data.frame(y=test$y[test$y  %in% c(2,3)], test$x[test$y  %in% c(2,3),])
length_divisor <- 100
train1 <- inTrain[,-1]
test1 <- inTest[,-1]
inTrain$y <- factor(inTrain$y)
numtrees <- c(25,50,100)
itr <- length(numtrees)
misclass3 <- vector('vector', length=itr)
j <- 1
for (i in numtrees) {
  misclasslst <- vector('numeric', length=i)
  rf1 <- randomForest(train1, inTrain$y, ntree=i )
  pt <- predict(rf1,inTest,predict.all=TRUE)
  for (t in 1:i) {
    tabs <- table(pt$individual[,t], inTest$y)
    misclasslst[t] <- 1 - sum(diag(tabs))/sum(tabs)
  }
  misclass3[[j]] <- misclasslst
  #if (j==1)  misclass3<-misclasso  else misclass3 <- c(misclass3,misclasso)
  j <- j+1
}
plot( rf1)

########################### Boosting ############################
inTrain = data.frame(y=train$y[train$y  %in% c(2,3)], train$x[train$y  %in% c(2,3),])
inTest = data.frame(y=test$y[test$y  %in% c(2,3)], test$x[test$y  %in% c(2,3),])
inTrain$y <- factor(inTrain$y)
itr <-50
misclass4 <- vector('numeric', length=itr)
misclass5 <- vector('numeric', length=itr)

library(ada)
#a
#default iter = 50
p4 <- ada(y ~ ., data = inTrain)
for (i in 1:itr) {
pt <- predict(p4,inTest,n.iter=i)
tabs <- table(pt, inTest$y)
misclass4[i] <- 1 - sum(diag(tabs))/sum(tabs)
}
p4
plot(p4)
#b
p5 <- ada(y ~ ., data = inTrain,control=rpart.control(maxdepth=2))
for (i in 1:itr) {
pt <- predict(p5,inTest,n.iter=i)
tabs <- table(pt, inTest$y)
misclass5[i] <- 1 - sum(diag(tabs))/sum(tabs)
}
plot(p5)

library(ggplot2)
library(reshape2)
x<-1:50
df <- data.frame(x,misClass,misClass1,misClass2[,1],misClass2[,2],misClass2[,3],misclass3[[2]],misclass4,misclass5)
names(df) <- c("x","NB Bag","Tree Bag","Tree Bag 10vr","Tree Bag 50vr", "Tree Bag 300vr","Random Forest 50tr","AdaBoosting","Ada maxdepth=2")
df2 <- melt(data = df, id.vars = "x")
ggplot(data = df2, aes(x = x, y = value, colour = variable)) + geom_line()

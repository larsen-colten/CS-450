# include LIBSVM
library (e1071)

#set working directroy
setwd("~/Documents/BYU-I/CS 450 - Machine Learning/SVM")

data <- read.csv(file = 'letters.csv', head=TRUE, sep=",")

data_allrows <- 1:nrow(data) 

data_testrows <- sample(data_allrows, length(data_allrows) *.3)

data_test <- data[data_testrows,]
data_train <- data[-data_testrows,]

model <- svm(letter~., data=data_train)

print(model)
summary(model)

predictions <- predict(model, data_test)
confusionMatrix <- table(pred = predictions, true = data_test$letter)

agreement <- predictions == data_test$letter
accuracy <- prop.table(table(agreement))

print(agreement)
print(accuracy)

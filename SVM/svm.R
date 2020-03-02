# install.packages('e1071', dependencies = TRUE)
# include LIBSVM
library (e1071)

#set working directroy
setwd("~/Documents/BYU-I/CS 450 - Machine Learning/SVM")

vowels <- read.csv(file = 'vowel.csv', head=TRUE, sep=",")
letters <- read.csv(file = 'letters.csv', head=TRUE, sep=",")

model_vowels <- svm(Class~., data=vowels)
model_letters <- svm(letter~., data=letters)
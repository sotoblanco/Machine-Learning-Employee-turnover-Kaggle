# Load libraries
library(tidyverse)
library(caret)
library(data.table)
library(Boruta)
library(skimr)
library(e1071)
library(rpart)
library(randomForest)

# set the columns name

columnNames <- c('experiene', # time 
                 'turnover', # employees' turnover 
                 'gender', # employees' gender
                 'age', # employees' age 
                 'industry', # employees' industry
                 'profession', # employees' profession
                 'traffic', # what pipeline employee came to the company
                 'coach', # presence of a coach (training) on probation
                 'head_gender', # gender of employees' boss (head)
                 'greywage', # employees' salary against minimum wage 
                 'way', # employees' way of transport
                 'extraversion', # employees' extraversion score
                 'independent', # employees' independent score 
                 'selfcontrol', # employees' self control score
                 'anxiety', # employees' anxiety score
                 'novator') # employees' novator score

list.files(path = "../Machine-Learning-Employee-turnover-Kaggle/data")


df <- read.csv("../Machine-Learning-Employee-turnover-Kaggle/data/turnover.csv",
               sep=',',
               header=TRUE,
               col.names=columnNames)


df <- subset(df, complete.cases(df))


df <- df %>% 
  mutate(turnover = as.character(turnover)) %>% 
  mutate(turnover = replace(turnover, turnover == '1', 'quit')) %>%
  mutate(turnover = replace(turnover, turnover == '0', 'no')) %>%
  mutate(turnover = as.factor(turnover)) %>%
  
  mutate(gender = replace(gender, gender == 'f', 'female')) %>%
  mutate(gender = replace(gender, gender == 'm', 'male')) %>%
  mutate(gender = as.factor(gender)) %>%
  
  mutate(head_gender = replace(head_gender, head_gender == 'f', 'female')) %>%
  mutate(head_gender = replace(head_gender, head_gender == 'm', 'male')) %>%
  mutate(head_gender = as.factor(head_gender))

head(df)

glimpse(df)

summary(df)

dim(df)

names(df)

skim(df)


na_count <-sapply(df, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
na_count


print(paste('There are', nrow(df), 'employees'))

df %>% 
  group_by(gender) %>%
  summarize(count= n())


# plot the gender distribution bar chart
df %>% ggplot(aes(x = gender)) + 
  geom_bar(fill = "blue", color = "black", width = 0.4) + ggtitle("Gender Distribution") + xlab("Gender") + ylab("Number of Employees")


# plot the profession distribution bar chart
df %>% ggplot(aes(x = profession)) + 
  geom_bar(fill = "blue", color = "black", width = 0.7) + ggtitle("Profession Distribution") +  xlab("Profession") + ylab("Number of Employees") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))



set.seed(100)
# partition the data with an 75:25 ratio split
trainIndex <- createDataPartition(df$turnover, p=.75, list=FALSE)
train_set <- df[trainIndex,]
test_set <- df[-trainIndex,]
head(train_set)


set.seed(100)
options(warn=-1)

subsets <- c(1:30)

# define the control function
ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

train_set <- train_set %>% select(-turnover, turnover)
test_set <- test_set %>% select(-turnover, turnover)


lmProfile <- rfe(x=train_set[,1:15], y=train_set$turnover,
                 sizes = 15,
                 rfeControl = ctrl)

lmProfile



boruta_output <- Boruta(turnover ~ ., data=train_set, doTrace=1) 


boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)

# Do a tentative rough fix
roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)


# Variable Importance Scores
imps <- attStats(roughFixMod)
imps2 = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
head(imps2[order(-imps2$meanImp), ])  # descending sort


# Plot variable importance
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")  


# remove the insignificant features from the train and test set
train_set <- train_set %>% subset(select = -c(head_gender, coach, novator, greywage, extraversion, gender))
test_set <- test_set %>% subset(select = -c(head_gender, coach, novator, greywage, extraversion, gender))


results <- data.frame(Model = character(), 
                      Accuracy = double(), 
                      Sensitivity = double(), 
                      Specificity = double(),
                      stringsAsFactors = FALSE)


knn_tec = train(turnover ~ ., data = train_set, method = "knn", preProcess=c('knnImpute'))
knn_tec


predictions = predict(knn_tec, newdata = test_set)
confusionMatrix <- confusionMatrix(predictions, test_set$turnover, positive='quit')
results[nrow(results) + 1, ] <- c(as.character('K-nearest neighbours (knn)'), 
                                  confusionMatrix$overall['Accuracy'],  
                                  confusionMatrix$byClass['Sensitivity'], 
                                  confusionMatrix$byClass['Specificity'])

confusionMatrix

fit <- naiveBayes(turnover ~ ., data = train_set)
predictions = predict(fit, test_set, type = 'class')
confusionMatrix <- confusionMatrix(predictions, test_set$turnover, positive='quit')
results[nrow(results) + 1, ] <- c(as.character('Naive Bayes'), 
                                  confusionMatrix$overall['Accuracy'],  
                                  confusionMatrix$byClass['Sensitivity'], 
                                  confusionMatrix$byClass['Specificity'])
confusionMatrix


fit <- rpart(turnover~., data = train_set, method = 'class')
predictions = predict(fit, test_set, type = 'class')
confusionMatrix <- confusionMatrix(predictions, test_set$turnover, prevalence = 0.06, positive='quit')
results[nrow(results) + 1, ] <- c(as.character('RPART'), 
                                  confusionMatrix$overall['Accuracy'],  
                                  confusionMatrix$byClass['Sensitivity'], 
                                  confusionMatrix$byClass['Specificity'])
confusionMatrix

fit <- train(turnover ~ .,  method = "pam", data = train_set)


predictions = predict(fit, test_set)
confusionMatrix <- confusionMatrix(predictions, test_set$turnover, positive='quit')
results[nrow(results) + 1, ] <- c(as.character('Partition Around Medoids'), 
                                  confusionMatrix$overall['Accuracy'],  
                                  confusionMatrix$byClass['Sensitivity'], 
                                  confusionMatrix$byClass['Specificity'])
confusionMatrix

set.seed(20)
rf <- randomForest(turnover ~ ., train_set, ntree=20)
predictions = predict(rf, test_set, type = 'class')
confusionMatrix <- confusionMatrix(predictions, test_set$turnover, prevalence = 0.06, positive='quit')
results[nrow(results) + 1, ] <- c(as.character('Random Forest'), 
                                  confusionMatrix$overall['Accuracy'],  
                                  confusionMatrix$byClass['Sensitivity'], 
                                  confusionMatrix$byClass['Specificity'])
confusionMatrix


results %>% knitr::kable(caption = "ACCURACY OF MACHINE LEARNING TECHNIQUES", label = NULL)
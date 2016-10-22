setwd("~/STAT642/Project")

library(readxl)
library(caret)
library(gbm)
library(pROC)
library(glmnet)
library(plyr)
library(e1071)
census <- read_excel("CensusFull.xlsx")

age <- census$Age
work <- census$Workclass
wgt <- census$fnlwgt
edu <- census$education
#edunum <- census$`Education-num`
marital <- census$`marital-status`
occu <- census$occupation
relat <- census$relationship
race <- census$race
sex <- census$sex
gain <- census$`capital-gain`
loss <- census$`capital-loss`
hours <- census$`hours-per-week`
native <- census$`native country`
salary <- census$salary

#creating dummy variables
work <- as.factor(work)
edu <- as.factor(edu)
marital <- as.factor(marital)
occu <- as.factor(occu)
relat <- as.factor(relat)
race <- as.factor(race)
sex <- as.factor(sex)
native <- as.factor(native)
censusdummy <- dummyVars("~.",data=census, fullRank=F)
census <- as.data.frame(predict(censusdummy,census))

#proportion of our outcome variables
prop.table(table(salary))
prop.table(table(relat))
prop.table(table(edu))

outcomeName <- 'salary'
predictorsNames <- names(census)[names(census) != outcomeName]

#########################################################
#########################################################
#use gbm by first creating a new classification variable
census$salary2 <- ifelse(salary==1,'yes','no')
census$salary2 <- as.factor(census$salary2)
outcomeName <- 'salary2'

#splitting into train and test data
set.seed(1234)
splitIndex <- createDataPartition(census[,outcomeName], p = .75, list = FALSE, times = 1)
trainDF <- census[ splitIndex,]
testDF  <- census[-splitIndex,]

#cross-validate
objControl <- trainControl(method='cv', number=10, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)
objModel <- train(trainDF[,predictorsNames], trainDF[,outcomeName], 
                  method='gbm', 
                  trControl=objControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"))

#find out what variables were most important:
as.matrix(summary(objModel))
#find out what tuning parameters were most important to the mode
print(objModel)

#find out accuracy of model
predictions <- predict(object=objModel, testDF[,predictorsNames], type='raw')
print(postResample(pred=predictions, obs=as.factor(testDF[,outcomeName])))

predictions <- predict(object=objModel, testDF[,predictorsNames], type='prob')
head(predictions)

#type 1, type 2 errors
confusionMatrix(objModel)

#area under curve (AUC)
auc <- roc(ifelse(testDF[,outcomeName]=="yes",1,0), predictions[[2]])
print(auc$auc)
plot(auc, main = "ROC Curve")

plot(objModel, main = "Performance per Iteration")


#########################################################
#########################################################
#GLMnet Modeling
outcomeName <- 'salary'

set.seed(1234)
splitIndex <- createDataPartition(census[,outcomeName], p = .75, list = FALSE, times = 1)
trainDF1 <- census[ splitIndex,]
testDF1  <- census[-splitIndex,]

objControl1 <- trainControl(method='cv', number=10, returnResamp='none')
objModel1 <- train(trainDF1[,predictorsNames], trainDF1[,outcomeName], method='glmnet',  metric = "RMSE", trControl=objControl1)

predictions1 <- predict(object=objModel1, testDF1[,predictorsNames])

auc1 <- roc(testDF1[,outcomeName], predictions1)
print(auc1$auc)
lines(auc1, col = "red")

plot(varImp(objModel1,scale=F))

xfactor <- trainDF[,-109]
xfactor <- data.frame(xfactor[,-109])
y <- trainDF$salary
glmnet(x=xfactor, y=as.factor(y), alpha=1, family='binomial')

x <- census[,-109]
x <- as.matrix(census[,-109])
y <- census[,-110]
y <- census$salary
lasso<-glmnet(x,y=y, alpha=1)

#########################################################
#########################################################
#logistic regression

Train <- createDataPartition(salary, p=0.75, list=FALSE)
training <- census[ Train, ]
testing <- census[ -Train, ]

exp(coef(mod_fit$finalModel))


mod_fit <- glm(training$salary~.,  data=census, family="binomial")
varImp(mod_fit, scale=F)

mod_fit1 <- glm(census$salary~.,  data=census)
varImp(mod_fit1, scale=T)



pred = predict(mod_fit, newdata=testing)
accuracy <- table(pred, testing[,"salary"])
sum(diag(accuracy))/sum(accuracy)



pred = predict(mod_fit, newdata=testing)
confusionMatrix(data=pred, testing$salary)


pred = predict(mod_fit, newdata=testing)
confusionMatrix(data=pred, testing$Class)




##
# decision tree
library(Deducer)
basemodel <- glm(salary~. - salary2, data=census)
confusionMatrix(basemodel)
rocplot(basemodel)
library(pROC)

library(party)
tree <- ctree(salary~., data=census)
tree <- ctree(basemodel)
plot(tree)

library(rpart)
fit <- rpart(salary~., data=census)
printcp(fit)

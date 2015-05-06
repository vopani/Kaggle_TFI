## setting working directory
path <- '/Kaggle_TFI'
setwd(path)


## loading libraries
library(caret)
library(dummies)
library(plyr)


## loading data (edit the paths)
train <- read.csv("./Raw/train.csv", stringsAsFactors=F)
test <- read.csv("./Raw/test.csv", stringsAsFactors=F)


## cleaning data
panel <- rbind(train[,-ncol(train)], test)

# creating feature variables
panel$year <- substr(as.character(panel$Open.Date),7,10)
panel$month <- substr(as.character(panel$Open.Date),1,2)
panel$day <- substr(as.character(panel$Open.Date),4,5)

panel$Date <- as.Date(strptime(panel$Open.Date, "%m/%d/%Y"))

panel$days <- as.numeric(as.Date("2014-02-02")-panel$Date)

panel$City.Group <- as.factor(panel$City.Group)

panel$Type[panel$Type == "DT"] <- "IL"
panel$Type[panel$Type == "MB"] <- "FC"
panel$Type <- as.factor(panel$Type)

panel <- subset(panel, select = -c(Open.Date, Date, City))

# converting some categorical variables into dummies
panel <- dummy.data.frame(panel, names=c("P14", "P15", "P16", "P17", "P18", "P19", "P20", "P21", "P22", "P23", "P24", "P25", "P30", "P31", "P32", "P33", "P34", "P35", "P36", "P37"), all=T)

ldf <- lapply(1:ncol(panel), function(k)
				{
					return(data.frame("column" = colnames(panel)[k],
									  "unique" = length(unique(panel[1:nrow(train),k]))))
				})

ldf <- ldply(ldf, data.frame)

# removing variables with unique values
panel <- panel[,!names(panel) %in% ldf$column[ldf$unique == 1]]

# removing highly correlated variables
for (i in (6:ncol(panel)))
{
	panel[,i] <- as.numeric(panel[,i])
}

cor <- cor(panel[1:nrow(train), 6:ncol(panel)])
high_cor <- findCorrelation(cor, cutoff = 0.99)

high_cor <- high_cor[high_cor != 186]

panel <- panel[,-c(high_cor+5)]

# splitting into train and test
X_train <- panel[1:nrow(train),-1]
X_test <- panel[(nrow(train)+1):nrow(panel),]

# building model on log of revenue
result <- log(train$revenue)


## Random Forest
source("./Codes/RandomForest_R.R")
model_rf_1 <- RandomForestRegression_CV(X_train,result,X_test,cv=5,ntree=25,nodesize=5,seed=235,metric="rmse")
model_rf_2 <- RandomForestRegression_CV(X_train,result,X_test,cv=5,ntree=25,nodesize=5,seed=357,metric="rmse")
model_rf_3 <- RandomForestRegression_CV(X_train,result,X_test,cv=5,ntree=25,nodesize=5,seed=13,metric="rmse")
model_rf_4 <- RandomForestRegression_CV(X_train,result,X_test,cv=5,ntree=25,nodesize=5,seed=753,metric="rmse")
model_rf_5 <- RandomForestRegression_CV(X_train,result,X_test,cv=5,ntree=25,nodesize=5,seed=532,metric="rmse")


## submission
test_rf_1 <- model_rf_1[[2]]
test_rf_2 <- model_rf_2[[2]]
test_rf_3 <- model_rf_3[[2]]
test_rf_4 <- model_rf_4[[2]]
test_rf_5 <- model_rf_5[[2]]

submit <- data.frame("Id" = test_rf_1$Id,
					 "Prediction" = 0.2*exp(test_rf_1$pred_rf) + 0.2*exp(test_rf_2$pred_rf) + 0.2*exp(test_rf_3$pred_rf) + 0.2*exp(test_rf_4$pred_rf) + 0.2*exp(test_rf_5$pred_rf))

write.csv(submit, "./Submission/submit.csv", row.names=F)


# loading library
suppressWarnings(library(pROC))
library(randomForest)


RandomForestClassification_CV <- function(X_train, y, X_test = data.frame(), cv = 5, ntree = 50, nodesize = 5, seed = 123, metric = "auc")
{
  score <- function(a,b,metric)
  {
    switch(metric,
           auc = auc(a,b),
           logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),
           precision = length(a[a==b])/length(a))
  }
  
  cat("Preparing Data\n")
  X_train$order <- seq(1, nrow(X_train))
  X_train$result <- as.factor(y)
  
  set.seed(seed)
  X_train$randomCV <- floor(runif(nrow(X_train), 1, (cv+1)))
  
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    X_build <- subset(X_train, randomCV != i, select = -c(order, randomCV))
    X_val <- subset(X_train, randomCV == i) 
    
    model_rf <- randomForest(result ~., data = X_build, ntree = ntree, nodesize = nodesize)
    
    varImpPlot(model_rf)
    #print(model_rf$importance)
    
    pred_rf <- predict(model_rf, X_val, type = "prob")[,2]
    X_val <- cbind(X_val, pred_rf)
    
    if (score == 'AUC')
    {
      X_val$pred_rf[X_val$pred_rf > 0.9999999999] <- 0.9999999999
      X_val$pred_rf[X_val$pred_rf < 0.0000000001] <- 0.0000000001
    }
    
    if (nrow(X_test) > 0)
    {
      pred_rf <- predict(model_rf, X_test, type = "prob")[,2]
    }
      
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_rf, metric), "\n", sep = "")
    
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, pred_rf)
      }      
    }
    
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_rf <- (X_test$pred_rf * (i-1) + pred_rf)/i
      }            
    }
    
    gc()
  } 

  output <- output[order(output$order),]
  cat("\nRandomForest ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_rf, metric), "\n", sep = "")

  output <- subset(output, select = c("order", "pred_rf"))
  return(list(output, X_test))  
}


RandomForestRegression_CV <- function(X_train, y, X_test = data.frame(), cv = 5, ntree = 50, nodesize = 5, seed = 123, metric = "mae")
{
  score <- function(a,b,metric)
  {
    switch(metric,
           mae = sum(abs(a-b))/length(a),
           rmse = sqrt(sum((a-b)^2)/length(a)))
  }
  
  cat("Preparing Data\n")
  X_train$order <- seq(1, nrow(X_train))
  X_train$result <- as.numeric(y)
  
  set.seed(seed)
  X_train$randomCV <- floor(runif(nrow(X_train), 1, (cv+1)))
  
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    X_build <- subset(X_train, randomCV != i, select = -c(order, randomCV))
    X_val <- subset(X_train, randomCV == i) 
    
    model_rf <- randomForest(result ~., data = X_build, ntree = ntree, nodesize = nodesize)
    
    varImpPlot(model_rf)
    #print(model_rf$importance)
    
    pred_rf <- predict(model_rf, X_val)
    X_val <- cbind(X_val, pred_rf)
    
    if (nrow(X_test) > 0)
    {
      pred_rf <- predict(model_rf, X_test)
    }
      
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_rf, metric), "\n", sep = "")
    
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, pred_rf)
      }      
    }
    
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_rf <- (X_test$pred_rf * (i-1) + pred_rf)/i
      }            
    }
    
    gc()
  } 

  output <- output[order(output$order),]
  cat("\nRandomForest ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_rf, metric), "\n", sep = "")

  output <- subset(output, select = c("order", "pred_rf"))
  return(list(output, X_test))  
}


## loading libraries
suppressWarnings(library(pROC))
library(randomForest)


RandomForestRegression_CV <- function(X_train,y,X_test=data.frame(),cv=5,ntree=50,nodesize=5,seed=123,metric="mae")
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


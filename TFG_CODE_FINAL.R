# Load required libraries
library(tidyverse)
library(caret)
library(e1071)
library(glmnet)  # For Lasso and Ridge regression
library(pROC)  # For ROC curve
library(randomForest)  # For Random Forest
library(corrplot) # For correlation heatmaps

# Load the dataset (already updated with numeric values)
data <- read.csv("C:\\Users\\ALBERTO M\\Downloads\\TFG_DATASET_MODEL.csv")

# Clean and convert each column to numeric, handling non-numeric characters
numeric_columns <- c("Income_Fluctuations", "Recurring_Expenses", "Cash_Flow_Stability",
                     "Utility_Payments_On_Time", "Rental_History_On_Time", "Subscription_Payments", 
                     "Spending_Patterns", "Real_Time_Transactions", "E_Commerce_Activity", 
                     "Local_Job_Market_Stability", "Inflation_Impact", "Economic_Downturn_Resilience",
                     "Employment_Sector", "Education_Level")

# Apply log transformation for skewed data (all values should be positive)
for (col in numeric_columns) {
  data[[col]] <- log1p(data[[col]])  # Apply log transformation (log(1+x) to avoid negative values)
}

# Round each numeric column to three decimal places
for (col in numeric_columns) {
  data[[col]] <- round(data[[col]], 3)  # Round to 3 decimal places
}

# Handle missing values (if any) before calculating Default_Probability
data[is.na(data)] <- 0  # Replace NAs with 0 for simplicity

# Recalculate Default_Probability with adjusted weights for each feature
set.seed(123)

data$Default_Probability <- 0.25 * data$Income_Fluctuations +  
  0.2 * data$Recurring_Expenses +  
  0.2 * data$Cash_Flow_Stability +  
  0.1 * data$Utility_Payments_On_Time +  
  0.1 * data$Rental_History_On_Time +  
  0.05 * data$Subscription_Payments +  
  0.05 * data$Spending_Patterns +  
  0.05 * data$Real_Time_Transactions +  
  0.05 * data$E_Commerce_Activity +  
  0.05 * data$Local_Job_Market_Stability +  
  0.05 * data$Inflation_Impact +  
  0.05 * data$Economic_Downturn_Resilience +  
  0.1 * data$Employment_Sector +  
  0.05 * data$Education_Level  

# Normalize to range [0, 100] for the default probability
data$Default_Probability <- (data$Default_Probability - min(data$Default_Probability)) / 
  (max(data$Default_Probability) - min(data$Default_Probability)) * 100

# Split the data into training (70%) and testing (30%) sets
data_clean <- data[complete.cases(data), ]  # Ensure no missing data for model training

trainIndex <- createDataPartition(data_clean$Default_Probability, p = 0.7, list = FALSE)
train_data <- data_clean[trainIndex, ]
test_data <- data_clean[-trainIndex, ]

# Scaling the data for Linear Regression and Random Forest
pre_process <- preProcess(train_data[, numeric_columns], method = c("center", "scale"))
train_data_scaled <- predict(pre_process, train_data)
test_data_scaled <- predict(pre_process, test_data)

# Lasso Regression (L1 regularization) with Cross-validation
lasso_grid <- expand.grid(alpha = 1, lambda = seq(0.0001, 1, length = 100))
cv_lasso <- train(Default_Probability ~ ., data = train_data_scaled, method = "glmnet", 
                  tuneGrid = lasso_grid, trControl = trainControl(method = "cv", number = 10))
lasso_preds <- predict(cv_lasso, test_data_scaled, type = "raw")

# Ridge Regression (L2 regularization) with Cross-validation
ridge_grid <- expand.grid(alpha = 0, lambda = seq(0.0001, 1, length = 100))
cv_ridge <- train(Default_Probability ~ ., data = train_data_scaled, method = "glmnet", 
                  tuneGrid = ridge_grid, trControl = trainControl(method = "cv", number = 10))
ridge_preds <- predict(cv_ridge, test_data_scaled, type = "raw")

# Random Forest Model with Hyperparameter Tuning
rf_model <- train(Default_Probability ~ ., data = train_data_scaled, method = "rf", 
                  tuneGrid = expand.grid(mtry = c(3, 4, 5)), 
                  trControl = trainControl(method = "cv", number = 10), 
                  ntree = 1000, nodesize = 5, importance = TRUE)  # Hyperparameters fixed
rf_preds <- predict(rf_model, test_data_scaled)

# --- Create the 'Decision_Default_Probability' Column --- #
threshold <- 50  # Set decision threshold
test_data$Decision_Default_Probability <- ifelse(test_data$Default_Probability >= threshold, "Yes", "No")

# --- Convert Default_Probability to binary using a threshold of 50 for ROC calculation ---
test_data$Binary_Default_Probability <- ifelse(test_data$Default_Probability >= threshold, "Yes", "No")

# --- Compute ROC Curves for Binary Classification --- #
roc_lasso <- roc(test_data$Binary_Default_Probability, lasso_preds, levels = c("No", "Yes"))
roc_ridge <- roc(test_data$Binary_Default_Probability, ridge_preds, levels = c("No", "Yes"))
roc_rf <- roc(test_data$Binary_Default_Probability, rf_preds, levels = c("No", "Yes"))

# Plot ROC curves
par(mfrow = c(1, 3))  # To display three plots side by side
plot(roc_lasso, main = "Lasso ROC", col = "blue")
plot(roc_ridge, main = "Ridge ROC", col = "red")
plot(roc_rf, main = "Random Forest ROC", col = "green")

# --- Evaluation Metrics --- #
rsq_lasso <- cor(test_data$Default_Probability, lasso_preds)^2
rmse_lasso <- sqrt(mean((lasso_preds - test_data$Default_Probability)^2))
mae_lasso <- mean(abs(lasso_preds - test_data$Default_Probability))

rsq_ridge <- cor(test_data$Default_Probability, ridge_preds)^2
rmse_ridge <- sqrt(mean((ridge_preds - test_data$Default_Probability)^2))
mae_ridge <- mean(abs(ridge_preds - test_data$Default_Probability))

rsq_rf <- cor(test_data$Default_Probability, rf_preds)^2
rmse_rf <- sqrt(mean((rf_preds - test_data$Default_Probability)^2))
mae_rf <- mean(abs(rf_preds - test_data$Default_Probability))

cat("Lasso Regression R-Squared: ", rsq_lasso, " RMSE: ", rmse_lasso, " MAE: ", mae_lasso, "\n")
cat("Ridge Regression R-Squared: ", rsq_ridge, " RMSE: ", rmse_ridge, " MAE: ", mae_ridge, "\n")
cat("Random Forest R-Squared: ", rsq_rf, " RMSE: ", rmse_rf, " MAE: ", mae_rf, "\n")

# --- Spearman Correlation --- #
spearman_lasso <- cor(test_data$Default_Probability, lasso_preds, method = "spearman")
spearman_ridge <- cor(test_data$Default_Probability, ridge_preds, method = "spearman")
spearman_rf <- cor(test_data$Default_Probability, rf_preds, method = "spearman")

# Plotting Spearman Correlations
library(ggplot2)
spearman_data <- data.frame(
  Model = c("Lasso", "Ridge", "Random Forest"),
  Spearman_Correlation = c(spearman_lasso, spearman_ridge, spearman_rf)
)

ggplot(spearman_data, aes(x = Model, y = Spearman_Correlation, fill = Model)) + 
  geom_bar(stat = "identity") +
  labs(title = "Spearman Correlation between Actual and Predicted Default Probabilities", 
       x = "Model", y = "Spearman Correlation") +
  theme_minimal()

# --- Correlation Matrix with Enhanced Plot --- #
cor_matrix <- cor(train_data_scaled[, numeric_columns])

# Plotting the Correlation Matrix using corrplot
library(corrplot)
corrplot(cor_matrix, 
         method = "color", 
         type = "upper", 
         order = "hclust", 
         col = colorRampPalette(c("red", "orange", "yellow"))(100), 
         title = "Correlation Matrix of Features", 
         tl.cex = 0.4, 
         cl.cex = 0.4, 
         tl.col = "black",  
         number.cex = 0.7,  
         addCoef.col = "black",  
         diag = FALSE)

# --- Feature Importance Plot (Random Forest) --- #
importance(rf_model$finalModel)  # Show feature importance
varImpPlot(rf_model$finalModel)  # Plot feature importance

# --- Summary Statistics for the Evaluation Metrics --- #
summary_stats <- data.frame(
  Model = c("Lasso", "Ridge", "Random Forest"),
  R_Squared = c(round(rsq_lasso, 4), round(rsq_ridge, 4), round(rsq_rf, 4)),
  RMSE = c(round(rmse_lasso, 4), round(rmse_ridge, 4), round(rmse_rf, 4)),
  MAE = c(round(mae_lasso, 4), round(mae_ridge, 4), round(mae_rf, 4)),
  AUC_Lasso = c(round(auc(roc_lasso), 4)),
  AUC_Ridge = c(round(auc(roc_ridge), 4)),
  AUC_RF = c(round(auc(roc_rf), 4)),
  Spearman_Lasso = c(round(spearman_lasso, 4)),
  Spearman_Ridge = c(round(spearman_ridge, 4)),
  Spearman_RF = c(round(spearman_rf, 4))
)

# Print the summary statistics table
print(summary_stats)

# --- Export the Final Results to CSV --- #
write.csv(test_data[, c("Default_Probability", "Decision_Default_Probability")], 
          "Credit_Risk_Assessment_Results_Final_Probabilities.csv", row.names = FALSE)

# Display the first few rows of the Default_Probability and Decision_Default_Probability columns
head(test_data[, c("Default_Probability", "Decision_Default_Probability")], n = 10)  # Display first 50 rows

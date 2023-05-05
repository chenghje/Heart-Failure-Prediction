# load required package
library(tidyverse)
library(tidymodels)
library(vip)
library(ranger)
library(xgboost)

### Data
df <- read_csv("heart.csv") %>% 
  mutate(Oldpeak=ifelse(Oldpeak<0, 0, Oldpeak), HeartDisease=factor(HeartDisease, levels=c("Yes", "No")))

# splitting
set.seed(123)
data_split <- initial_split(df, prop = 0.7, strata = "HeartDisease")
train_data <- training(data_split)
test_data <- testing(data_split)

# recipe 
fit_recipe <- recipe(HeartDisease ~ ., data = train_data) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_normalize(all_numeric()) 

# processed
fit_processed <- fit_recipe %>% prep() %>% juice()

# cross-validation
fit_cv <- vfold_cv(train_data, v = 5)





### Ranger Model
# model specification
rf_spec <- rand_forest(mtry=tune(), min_n=tune(), trees=tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification") 

# grid
rf_grid <- grid_regular(
  mtry(c(1, 5)), min_n(c(1, 10)), trees(c(100, 1000)), levels = 5)

# Workflow
rf_workflow <- workflow() %>% add_recipe(fit_recipe) %>% add_model(rf_spec)

# tuning
set.seed(123) 
rf_tuned <- rf_workflow %>% 
  tune_grid(
    resamples = fit_cv,
    grid = rf_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(verbose = F, save_pred = TRUE)
  )

# collect results
autoplot(rf_tuned)
collect_metrics(rf_tuned)
show_best(rf_tuned, "roc_auc")

# best parameters
rf_best <- rf_tuned %>% select_best("roc_auc") 

# final model
final_rf <- finalize_workflow(rf_workflow, rf_best) %>% fit(data = train_data)  

# train importance
final_rf %>% extract_fit_parsnip() %>% vip()

# train roc
rf_tuned %>% collect_predictions(parameters = rf_best) %>% roc_curve(HeartDisease, .pred_Yes) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(linewidth = 1.5, color = "midnightblue") +
  geom_abline(lty = 2, alpha = 0.5, color = "gray50", linewidth = 1.2) +
  labs(title="ROC")

rf_tuned %>% collect_predictions(parameters = rf_best) %>% roc_auc(HeartDisease, .pred_Yes) 

# test prediction 
rf_predictions <- final_rf %>% predict(test_data) %>% bind_cols(test_data)

# test evaluation
rf_predictions %>% metrics(truth = HeartDisease, estimate = .pred_class)

# test importance
final_rf %>% last_fit(data_split) %>% extract_fit_parsnip() %>% vip()

# test roc
final_rf %>% last_fit(data_split) %>% collect_predictions() %>% roc_curve(HeartDisease, .pred_Yes) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(linewidth = 1.5, color = "midnightblue") +
  geom_abline(lty = 2, alpha = 0.5, color = "gray50", linewidth = 1.2) +
  labs(title="ROC")

final_rf %>% last_fit(data_split) %>% collect_predictions() %>% roc_auc(HeartDisease, .pred_Yes)






### XGB Model
# model specification
xgb_spec <- boost_tree(trees = 500,
  tree_depth = tune(), min_n = tune(), loss_reduction = tune(),    
  sample_size = tune(), mtry = tune(), learn_rate = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification") 

# grid
xgb_grid <- grid_latin_hypercube(
  mtry(c(1, 5)), tree_depth(c(3, 10)), min_n(c(1, 14)), 
  learn_rate(), loss_reduction(), sample_size = sample_prop(), 
  size = 40
)

# Workflow
xgb_workflow <- workflow() %>% add_recipe(fit_recipe) %>% add_model(xgb_spec)

# tuning
set.seed(123) 
xgb_tuned <- xgb_workflow %>% 
  tune_grid(
    resamples = fit_cv,
    grid = xgb_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(verbose = F, save_pred = TRUE)
  )

# collect results
autoplot(rf_tuned)
collect_metrics(xgb_tuned)
show_best(xgb_tuned, "roc_auc")

# best parameters
xgb_best <- xgb_tuned %>% select_best("roc_auc") 

# final model
final_xgb <- finalize_workflow(xgb_workflow, xgb_best) %>% fit(data = train_data)  

# train importance
final_xgb %>% extract_fit_parsnip() %>% vip()

# train roc
xgb_tuned %>% collect_predictions(parameters = xgb_best) %>% roc_curve(HeartDisease, .pred_Yes) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(linewidth = 1.5, color = "midnightblue") +
  geom_abline(lty = 2, alpha = 0.5, color = "gray50", linewidth = 1.2) +
  labs(title="ROC")

xgb_tuned %>% collect_predictions(parameters = xgb_best) %>% roc_auc(HeartDisease, .pred_Yes)

# test prediction 
xgb_predictions <- final_xgb %>% predict(test_data) %>% bind_cols(test_data)

# test evaluation
xgb_predictions %>% metrics(truth = HeartDisease, estimate = .pred_class)

# test importance
final_xgb %>% last_fit(data_split) %>% extract_fit_parsnip() %>% vip()

# test roc
final_xgb %>% last_fit(data_split) %>% collect_predictions() %>% roc_curve(HeartDisease, .pred_Yes) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(linewidth = 1.5, color = "midnightblue") +
  geom_abline(lty = 2, alpha = 0.5, color = "gray50", linewidth = 1.2) +
  labs(title="ROC")

final_xgb %>% last_fit(data_split) %>% collect_predictions() %>% roc_auc(HeartDisease, .pred_Yes)






# Save Better Rand Model
save(final_rf, file = "rf_model_tidym.Rdata")

final_rf %>% extract_fit_parsnip() %>% vip(num_features=30) + title("Variable Importance Weight")


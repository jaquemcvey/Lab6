library(tidyverse)
library(tidymodels)
library(glue)
library(baguette)

##  **Q1**: Download data (10 pts)
root <- 'https://gdex.ucar.edu/dataset/camels/file'
remote_files  <- glue('{root}/camels_{types}.txt')
local_files   <- glue('labs/data/camels_{types}.txt')

types <- c("clim", "geol", "soil", "topo", "vege", "hydro")
# PDF
download.file(glue('{root}/camels_attributes_v2.0.pdf'), 
              'labs/data/camels_attributes_v2.0.pdf')
# Get files
walk2(remote_files, local_files, download.file, quiet = TRUE)
# Open
camels <- map(local_files, read_delim, show_col_types = FALSE) |> 
  powerjoin::power_full_join() 

# > zero_q_freq: frequency of days with Q = 0 mm/day

# **Q2**: Make 2 maps (10 pts)

ggplot(data = camels, aes(x = gauge_lon, y = gauge_lat)) +
  borders("state", colour = "gray50") +
  geom_point(aes(color = aridity)) +
  scale_color_gradient(low = "darkgreen", high = "brown") +
  ggthemes::theme_map()

ggplot(data = camels, aes(x = gauge_lon, y = gauge_lat)) +
  borders("state", colour = "gray50") +
  geom_point(aes(color = p_mean)) +
  scale_color_gradient(low = "gray", high = "dodgerblue") +
  ggthemes::theme_map()

## **Q3**: Build a xgboost and neural network model (20 pts)
## 
## --- FROM LAB --- #
## 
rec <-  recipe(logQmean ~ aridity + p_mean, data = camels_train) %>%
  # Log transform the predictor variables (aridity and p_mean)
  step_log(all_predictors()) %>%
  # Add an interaction term between aridity and p_mean
  step_interact(terms = ~ aridity:p_mean) |> 
  # Drop any rows with missing values in the pred
  step_naomit(all_predictors(), all_outcomes())

set.seed(123)
# Bad form to perform simple transformations on the outcome variable within a 
# recipe. So, we'll do it here.
camels <- camels |> 
  mutate(logQmean = log(q_mean))

# Generate the split
camels_split <- initial_split(camels, prop = 0.8)
camels_train <- training(camels_split)
camels_test  <- testing(camels_split)

camels_cv <- vfold_cv(camels_train, v = 10)
### --- END FROM LAB --- ###

xgb <- boost_tree(mode = "regression", trees = 1000)

nn <- bag_mlp(mode = "regression") %>%
  set_engine("nnet")

xgb_workflow <- workflow() %>%
  add_model(xgb) %>% 
  add_recipe(rec) %>%
  fit(data = camels_train) |> 
  augment(camels_train)

nn_workflow <- workflow() %>%
  add_model(nn) %>%
  add_recipe(rec) %>%
  fit(data = camels_train)

## Q4: Chose you adventure (one option!!)
## **Q4a**: Data Prep / Data Splitting (15)
camels2 <- camels %>%
  mutate(logQmean = log(q_mean)) %>%
  select(logQmean, p_mean, aridity, soil_depth_pelletier, max_water_content, organic_frac, frac_snow, pet_mean, soil_depth_statsgo, elev_mean) %>%
  na.omit()

set.seed(1991)
c_split <-  initial_split(camels2, prop = 0.8)
c_train <-  training(c_split)
c_test  <-  testing(c_split)
cv <-  vfold_cv(c_train, v = 10)

## **Q4b**: Recipe (15)
rec <-  recipe(logQmean ~ ., data = c_train) %>%
  step_scale(all_predictors()) %>%
  step_center(all_predictors())

## **Q4c**: Define 3 models (25)
dt <- decision_tree(mode = "regression") %>%
  set_engine("rpart")

xgb <- boost_tree(mode = "regression", trees = 1000)

rf <- rand_forest(mode = "regression") %>%
  set_engine("ranger")

nn <- bag_mlp(mode = "regression") %>%
  set_engine("nnet")

# **Q4d**: workflow set (15)
wf_obj <- workflow_set(list(rec), list(rf, nn, xgb, dt)) %>%
  workflow_map("fit_resamples", 
               resamples = cv,
               control = control_resamples(save_pred = TRUE)) 

# **Q4e**: Evaluation (15)
autoplot(wf_obj)
rank_results(wf_obj, rank_metric = "rmse")

# **Q4f**: Extract and Evaluate (15)

m <- workflow() %>%
  add_model(nn) %>%
  add_recipe(rec) %>%
  fit(data = c_train)

a <- augment(m, c_test) 

ggplot(a, aes(x = logQmean, y = .pred)) +
  geom_point() +
  geom_abline() +
  labs(title = "Neural Network Model", x = "Truth", y = "Prediction") +
  theme_minimal()
  
metrics(a, truth = logQmean, estimate = .pred) 

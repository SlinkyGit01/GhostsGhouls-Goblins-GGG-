##################################### GGG ######################################



library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)

setwd("~/School/F2023/STAT348/STAT348/GhostsGhouls-Goblins-GGG-")



####################################### EDA ####################################



train <- vroom("train.csv")

test <- vroom("test.csv")

# gtNA <- vroom("trainWithMissingValues.csv")
# 
# my_recipe <- recipe(type ~ ., data=gtNA) %>%
#   step_impute_median(bone_length) %>% 
#   step_impute_median(hair_length)
# 
# prep <- prep(my_recipe)
# baked <- bake(prep, new_data = gtNA)
# 
# rmse_vec(gt[is.na(gtNA)], baked[is.na(gtNA)])



################################### SVM ########################################



library(doParallel)

num_cores <- parallel::detectCores()

cl <- makePSOCKcluster(num_cores)

registerDoParallel(cl)

train <- vroom("train.csv")
test <- vroom("test.csv")
train$type <- as.factor(train$type)

my_recipe <- recipe(type ~ ., data = train) %>%
  #update_role(id, new_role = "ID") %>%
  #step_dummy(color)  # dummy variable encoding
  step_lencode_glm(color, outcome = vars(type))

bake(prep(my_recipe), new_data = train)

svm_mod <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svm_mod)

svm_tuning_grid <- grid_regular(rbf_sigma(), cost(),levels = 15)

folds <- vfold_cv(train, v = 15, repeats=1)

svm_results <- svm_wf %>%
  tune_grid(resamples = folds,
            grid = svm_tuning_grid,
            metrics = metric_set(accuracy))

svm_bestTune <- svm_results %>%
  select_best("accuracy")

svm_final_wf <- svm_wf %>%
  finalize_workflow(svm_bestTune) %>%
  fit(data=train)

svm_preds <- predict(svm_final_wf,
                     new_data=test,
                     type="class")

svm_submit <- as.data.frame(cbind(test$id, as.character(svm_preds$.pred_class)))
colnames(svm_submit) <- c("id", "type")
write_csv(svm_submit, "svm_submit.csv")

stopCluster(cl)



################################# Neural Networks ##############################



library(keras)
library(tensorflow)

nn_recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role="ID") %>%
  step_lencode_glm(color, outcome = vars(type)) %>% 
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_mod <- mlp(hidden_units = tune(),
                epochs = 50) %>%
  set_engine("nnet") %>% #verbose = 0 prints off less
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_mod)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 2)),
                            levels = 30)

folds <- vfold_cv(train, v = 5, repeats=1)

nn_results <- nn_wf %>%
  tune_grid(resamples = folds,
            grid = nn_tuneGrid,
            metrics = metric_set(accuracy))

nn_bestTune <- nn_results %>%
  select_best("accuracy")

nn_final_wf <- nn_wf %>%
  finalize_workflow(nn_bestTune) %>%
  fit(data=train)

nn_preds <- predict(nn_final_wf,
                     new_data=test,
                     type="class")

nn_submit <- as.data.frame(cbind(test$id, as.character(nn_preds$.pred_class)))
colnames(nn_submit) <- c("id", "type")
write_csv(nn_submit, "nn_submit.csv")

nn_results %>% collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

## CV tune, finalize and predict here and save results
## This takes a few min (10 on my laptop) so run it on becker if you want



################################ Boosting/Bart #################################



library(doParallel)

num_cores <- parallel::detectCores()

cl <- makePSOCKcluster(num_cores)

registerDoParallel(cl)

library(bonsai)
library(lightgbm)

train$type <- as.factor(train$type)

my_recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role = "ID") %>%
  #step_dummy(color)  # dummy variable encoding
  step_lencode_glm(color, outcome = vars(type))

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

## Set up grid of tuning values

boostGrid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 3)

## Set up K-fold CV

folds <- vfold_cv(train, v = 3, repeats=1)

boost_results <- boost_wf %>%
  tune_grid(resamples = folds,
            grid = boostGrid,
            metrics = metric_set(accuracy))

boost_bestTune <- boost_results %>%
  select_best("accuracy")

boost_final_wf <- boost_wf %>%
  finalize_workflow(boost_bestTune) %>%
  fit(data=train)

boost_preds <- predict(boost_final_wf,
                    new_data=test,
                    type="class")

boost_submit <- as.data.frame(cbind(test$id, as.character(boost_preds$.pred_class)))
colnames(boost_submit) <- c("id", "type")
write_csv(boost_submit, "boost_submit.csv")
## CV tune, finalize and predict here and save results

stopCluster(cl)

############################### Bart ###########################################



library(bonsai)
library(lightgbm)

library(doParallel)

num_cores <- parallel::detectCores()

cl <- makePSOCKcluster(num_cores)

registerDoParallel(cl)

train$type <- as.factor(train$type)

my_recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role = "ID") %>%
  #step_dummy(color)  # dummy variable encoding
  step_lencode_glm(color, outcome = vars(type))

bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)

## Set up grid of tuning values

bartGrid <- grid_regular(trees(),
                          levels = 3)

## Set up K-fold CV

folds <- vfold_cv(train, v = 3, repeats=1)

bart_results <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = bartGrid,
            metrics = metric_set(accuracy))

bart_bestTune <- bart_results %>%
  select_best("accuracy")

bart_final_wf <- bart_wf %>%
  finalize_workflow(bart_bestTune) %>%
  fit(data=train)

bart_preds <- predict(bart_final_wf,
                       new_data=test,
                       type="class")

bart_submit <- as.data.frame(cbind(test$id, as.character(bart_preds$.pred_class)))
colnames(bart_submit) <- c("id", "type")
write_csv(bart_submit, "bart_submit.csv")

stopCluster(cl)



##################################### N Bayes ##################################



library(discrim)
library(naivebayes)

## Recipe

train$type <- as.factor(train$type)

my_recipe <- recipe(type ~ ., data = train) %>%
  #update_role(id, new_role = "ID") %>%
  #step_dummy(color)  # dummy variable encoding
  step_lencode_glm(color, outcome = vars(type))

## model and workflow

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naive bayes eng

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here

## Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL

# Find Best Tuning Parameters1
best_tune_nb <- CV_results %>%
  select_best("accuracy")

## Finalize the Workflow & fit it
final_wf <- nb_wf %>%
  finalize_workflow(best_tune_nb) %>%
  fit(data=train)


## Predict

#predict(final_wf, new_data=atest, type="prob")

ggg_nb <- final_wf %>% predict(new_data=test,
                                              type="class")

nb_submit <- as.data.frame(cbind(test$id, as.character(ggg_nb$.pred_class)))
colnames(nb_submit) <- c("id", "type")
write_csv(nb_submit, "nb_submit.csv")




















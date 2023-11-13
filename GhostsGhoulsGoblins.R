##################################### GGG ######################################



library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)

setwd("~/School/F2023/STAT348/STAT348/GhostsGhouls-Goblins-GGG-")



####################################### EDA ####################################



gt <- vroom("train.csv")

gtest <- vroom("test.csv")

gtNA <- vroom("trainWithMissingValues.csv")

my_recipe <- recipe(type ~ ., data=gtNA) %>%
  step_impute_median(bone_length) %>% 
  step_impute_median(hair_length)

prep <- prep(my_recipe)
baked <- bake(prep, new_data = gtNA)

rmse_vec(gt[is.na(gtNA)], baked[is.na(gtNA)])

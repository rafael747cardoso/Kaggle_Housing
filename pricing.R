
############################################ House Pricing Regression ############################################

require(ggplot2)
require(gridExtra)
require(plotly)
require(dplyr)
require(tidyr)
require(fastDummies)
require(boot)
require(glmnet)
require(pls)

source("./funcs/fix_bad_levels.R")
source("./funcs/make_dens_plot.R")
source("./funcs/make_boxplot.R")
source("./funcs/make_shrinkage_plot.R")
source("./funcs/make_subset_selection_plot.R")

set.seed(666)

############ Training data

# Read:
df = read.csv("./data/train.csv")

### Preprocessing

# Fix the levels:
cat_vars = 


df$MSZoning = fix_bad_levels(df$MSZoning)

sort(unique(df$MSZoning))




# To each model:
df_train_forward = df
df_train_ridge = df
df_train_lasso = df
df_train_pcr = df
df_train_pls = df


############ Model selection

###### Forward Stepwise Selection

###### Ridge Regression

###### The Lasso

###### Principal Components Regression

###### Partial Least Squares

###### Comparison

# Plot:

# Best model:
# best_fit = fit_ ...



############ Test data

# Read:
df_test = read.csv("./data/test.csv")

### Preprocessing




# Prediction:
# y_pred = predict(best_fit,
#                  newx = df_test)
# df_pred = data.frame(
#     "Id" = df_test$Id,
#     "SalePrice" = y_pred
# )
# write.csv(object = df_pred,
#           file = "./data/prediction.csv",
#           sep = ",",
#           row.names = FALSE)




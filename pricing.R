
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
require(Amelia)

source("./funcs/fix_bad_levels.R")
source("./funcs/make_dens_plot.R")
source("./funcs/make_boxplot.R")
source("./funcs/make_shrinkage_plot.R")
source("./funcs/make_subset_selection_plot.R")

set.seed(666)

############ Training data

# Read:
df_train = read.csv("./data/train.csv")
df_test = read.csv("./data/test.csv")

### Preprocessing

# Concatenate the training and test sets to maintain the structure:
df_test$SalePrice = -1 # just to know it is the test set
df_all = rbind(df_train,
               df_test)

# Fix the variables names starting with a number:
vars_names = names(df_all)
vars_names[which(vars_names == "X1stFlrSF")] = "FirstFlrSF"
vars_names[which(vars_names == "X2ndFlrSF")] = "SecondFlrSF"
vars_names[which(vars_names == "X3SsnPorch")] = "ThreeSsnPorch"
names(df_all) = vars_names

# Types of variables:
response_var = "SalePrice"
id_var = "Id"
cat_vars = df_all %>%
               dplyr::select_if(~!is.numeric(.)) %>%
               names()
num_vars = names(df_all)[which(!(names(df_all) %in% c(response_var, cat_vars, id_var)))]

# Remove variables with too many NA's (more than 5% of the observations):
df_na = df_all %>%
            summarise(across(everything(), ~ sum(is.na(.)))) %>%
            t()
df_na = data.frame(
    "var_name" = row.names(df_na),
    "NAs" = df_na[, 1],
    stringsAsFactors = FALSE
) %>%
    dplyr::filter(NAs < 0.05*nrow(df_all))
df_all = df_all %>%
              dplyr::select(all_of(df_na$var_name))

# Replace the remaining NA's with the mean (for numeric) or the mode (for categoric):
df_all


# Fix the levels:
for(var in cat_vars){
    df_all[var] = fix_bad_levels(df_all[var])
}

# Dummies:
df = fastDummies::dummy_cols(.data = df,
                             select_columns = cat_vars,
                             remove_selected_columns = TRUE,
                             remove_first_dummy = TRUE)
dummy_vars = names(df)[which(apply(X = df,
                                   MARGIN = 2,
                                   FUN = function(x) length(unique(x))) == 2)]




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



############ Prediction

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




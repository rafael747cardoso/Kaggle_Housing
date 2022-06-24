
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
source("./funcs/is_dummy.R")
source("./funcs/getmode.R")
source("./funcs/make_dens_plot.R")
source("./funcs/make_boxplot.R")
source("./funcs/make_shrinkage_plot.R")
source("./funcs/make_subset_selection_plot.R")

set.seed(666)

############ Data

# Read:
df_train = read.csv("./data/train.csv")
df_test = read.csv("./data/test.csv")

### Preprocessing

# Special variables:
response_var = "SalePrice"
id_var = "Id"

# Concatenate the training and test sets to maintain the structure:
df_test[response_var] = -1 # just to know it is the test set
df_all = rbind(df_train,
               df_test)

# Fix the variable names starting with a number:
vars_names = names(df_all)
vars_names[which(vars_names == "X1stFlrSF")] = "FirstFlrSF"
vars_names[which(vars_names == "X2ndFlrSF")] = "SecondFlrSF"
vars_names[which(vars_names == "X3SsnPorch")] = "ThreeSsnPorch"
names(df_all) = vars_names

# Remove variables with too many NA's (more than 5% of the observations) in each column:
Amelia::missmap(df_all)
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
Amelia::missmap(df_all)

# Remove variables with only 1 value:
df_one = df_all %>%
            dplyr::summarise(across(everything(), ~ length(unique(.)))) %>%
            t()
df_one = data.frame(
    "var_name" = row.names(df_one),
    "len_uniques" = df_one[, 1],
    stringsAsFactors = FALSE
) %>%
    dplyr::filter(len_uniques > 1)
df_all = df_all %>%
              dplyr::select(all_of(df_one$var_name))
Amelia::missmap(df_all)

# Types of variables:
cat_vars = df_all %>%
               dplyr::select_if(~!is.numeric(.)) %>%
               names()
num_vars = names(df_all)[which(!(names(df_all) %in% c(response_var, cat_vars, id_var)))]

# Replace the remaining NA's with the mean (for numeric) or the mode (for categoric) in each column:
for(var in num_vars){
    if(sum(is.na(df_all[var])) > 0){
        df_all[which(is.na(df_all[var])), var] = mean(df_all[, var], na.rm = TRUE)
    }
}
for(var in cat_vars){
    if(sum(is.na(df_all[var])) > 0){
        df_all[is.na(df_all[var]), var] = getmode(df_all[, var])
    }
}
Amelia::missmap(df_all)

# Fix the levels:
for(var in cat_vars){
    df_all[, var] = fix_bad_levels(df_all[, var])
}

# Dummies:
df_all = fastDummies::dummy_cols(.data = df_all,
                                 select_columns = cat_vars,
                                 remove_selected_columns = TRUE,
                                 remove_first_dummy = TRUE)
dummy_vars = names(df_all)[apply(X = df_all,
                                 MARGIN = 2,
                                 FUN = is_dummy)]

# Back to the train/test split:
df_train = df_all %>%
               dplyr::filter(all_of(response_var) > -1)
df_test = df_all %>%
              dplyr::filter(eval(parse(text = response_var)) == -1) %>%
              dplyr::select(-all_of(response_var))

# Standardize the predictors from training and test:
predictors = names(df_train)[!(names(df_train) %in% c(response_var, id_var))]
predictors_not_dummies = predictors[!(predictors %in% dummy_vars)]
predictors_not_dummies = predictors_not_dummies[!grepl(x = predictors_not_dummies,
                                                       pattern = "_")] # for this, variables must be in camelCase
X_means_train = apply(X = df_train[predictors_not_dummies],
                      MARGIN = 2,
                      FUN = function(x) mean(x))
X_sd_train = apply(X = df_train[predictors_not_dummies],
                   MARGIN = 2,
                   FUN = function(x) sd(x))


standardise_x = function(x){
    x_stand = (x - X_means_train)/X_sd_train
    
}

df_train_stand = data.frame(
    id_var = df_train[, id_var],
    apply(X = df_train[predictors_not_dummies],
          MARGIN = 2,
          FUN = standardise_x
          # FUN = function(x) (x - mean(x)  )/sd(x)
          ),
    df_train[, dummy_vars],
    response_var = df_train[, response_var]
)
names(df_train_stand)[which(names(df_train_stand) == "id_var")] = id_var
names(df_train_stand)[which(names(df_train_stand) == "response_var")] = response_var




# Outliers:
get_outliers = function(x){
   which(x > quantile(x)[4] + 50*IQR(x) |
         x < quantile(x)[2] - 10*IQR(x))
}
ind_outs = c()
df_predictors = df_train %>%
                    dplyr::select(-all_of(dummy_vars),
                                  -all_of(response_var),
                                  -all_of(id_var))
for(j in 1:ncol(df_predictors)){
    ind_outs = c(ind_outs,
                 get_outliers(df_predictors[, j]))
    print(length(get_outliers(df_predictors[, j])))
}
ind_outs = unique(ind_outs)
df_train = df_train[-ind_outs, ]





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





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
source("./funcs/multi_reg_plots.R")

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
               dplyr::filter(eval(parse(text = response_var)) > -1)
df_test = df_all %>%
              dplyr::filter(eval(parse(text = response_var)) == -1) %>%
              dplyr::select(-all_of(response_var))

### Standardize the predictors from training and test

# For this, the variables must be in camelCase:
predictors = names(df_train)[!(names(df_train) %in% c(response_var, id_var))]
predictors_not_dummies = predictors[!(predictors %in% dummy_vars)]
predictors_not_dummies = predictors_not_dummies[!grepl(x = predictors_not_dummies,
                                                       pattern = "_")]

# Training:
df_train_prdnotdum = df_train[predictors_not_dummies]
X_means_train = c()
X_sd_train = c()
for(j in 1:ncol(df_train_prdnotdum)){
    x = df_train_prdnotdum[, j]
    X_means_train = c(X_means_train,
                      mean(x))
    X_sd_train = c(X_sd_train,
                   sd(x))
    df_train_prdnotdum[, j] = (x - X_means_train[j])/X_sd_train[j]
}
df_train_stand = data.frame(
    id_var = df_train[, id_var],
    df_train_prdnotdum,
    df_train[, dummy_vars],
    response_var = df_train[, response_var]
)
names(df_train_stand)[which(names(df_train_stand) == "id_var")] = id_var
names(df_train_stand)[which(names(df_train_stand) == "response_var")] = response_var

# Test:
df_test_prdnotdum = df_test[predictors_not_dummies]
for(j in 1:ncol(df_test_prdnotdum)){
    x = df_test_prdnotdum[, j]
    df_test_prdnotdum[, j] = (x - X_means_train[j])/X_sd_train[j]
}
df_test_stand = data.frame(
    id_var = df_test[, id_var],
    df_test_prdnotdum,
    df_test[, dummy_vars]
)
names(df_test_stand)[which(names(df_test_stand) == "id_var")] = id_var

### Outliers

## maybe remove the predictors with unatural peaks before removing outliers automatically
# get_outliers = function(x){
#    which(x > quantile(x)[4] + 1000*IQR(x) |
#          x < quantile(x)[2] - 100*IQR(x))
# }
# ind_outs = c()
# df_predictors = df_train_stand %>%
#                     dplyr::select(-all_of(dummy_vars),
#                                   -all_of(response_var),
#                                   -all_of(id_var))
# for(j in 1:ncol(df_predictors)){
#     ind_outs = c(ind_outs,
#                  get_outliers(df_predictors[, j]))
#     print(length(get_outliers(df_predictors[, j])))
# }
# ind_outs = unique(ind_outs)
# df_train_stand = df_train_stand[-ind_outs, ]
# boxplot(df_predictors)

# To each model:
df_train_stand = df_train_stand %>%
                     dplyr::select(-all_of(id_var))
df_train_forward = df_train_stand
df_train_ridge = df_train_stand
df_train_lasso = df_train_stand
df_train_pcr = df_train_stand
df_train_pls = df_train_stand

############ Model selection

###### Forward Stepwise Selection

# A generic multiple linear regression model:
cv_mse_mlr = function(predictors, df_model){
    # Fit:
    df_model = df_model[, c(response_var, predictors)]
    fit = glm(formula = SalePrice ~ .,
              data = df_model)
    
    # Cross-validated MSE:
    cv_mse = boot::cv.glm(data = df_model,
                          glmfit = fit,
                          K = 10)$delta[1]
    return(cv_mse)
}

### K-Fold Cross-validated MSE

p_predictors = names(df_train_forward)[names(df_train_forward) != response_var]
cv_mse = c()
cv_mse_se = c()
num_predictors = c()
predictors_names = c()
p = length(p_predictors)

# Null model:
fit = glm(formula = eval(parse(text = response_var)) ~ 1,
          data = df_train_forward)
cv_mse_null = boot::cv.glm(data = df_train_forward,
                           glmfit = fit,
                           K = 10)$delta[1]
cv_mse = c(cv_mse,
           cv_mse_null)
cv_mse_se = c(cv_mse_se,
              sd(cv_mse_null))
num_predictors = c(num_predictors,
                   0)
predictors_names = c(predictors_names,
                     "")

# Forward selection:
used_predictors = c()
for(k in 0:(p - 1)){
    print(p - 1 - k)
    # The p - k models that augment the predictors in one:
    cv_mse_k = c()
    predictors_k = c()
    available_predictors = p_predictors[!(p_predictors %in% used_predictors)]
    for(j in 1:length(available_predictors)){
        additional_predictor = available_predictors[j]
        cv_mse_kj = cv_mse_mlr(predictors = c(used_predictors,
                                              additional_predictor),
                               df_model = df_train_forward)
        cv_mse_k = c(cv_mse_k,
                     cv_mse_kj)
        predictors_k = c(predictors_k,
                         additional_predictor)
    }
    
    # Choose the best submodel:
    chosen_predictor = predictors_k[which(cv_mse_k == min(cv_mse_k))]
    used_predictors = c(used_predictors,
                        chosen_predictor)
    cv_mse = c(cv_mse,
               min(cv_mse_k))
    cv_mse_se = c(cv_mse_se,
                  sd(cv_mse_k)/sqrt(nrow(df_train_forward)))
    num_predictors = c(num_predictors,
                       k + 1)
    predictors_names = c(predictors_names,
                         paste(used_predictors,
                               collapse = ","))
}

# Prediction error values:
df_eval = data.frame(
    "num_predictors" = num_predictors,
    "cv_mse" = cv_mse,
    "cv_mse_se" = cv_mse_se,
    "predictors" = predictors_names
)
df_eval$cv_mse_se[is.na(df_eval$cv_mse_se)] = 0

# Best model with the 1-standard-error rule:
min_cv_mse = min(df_eval$cv_mse)
for(i in 2:nrow(df_eval)){
    if(df_eval$cv_mse[i] - df_eval$cv_mse_se[i] <= min_cv_mse){
        best_p = i - 1
        break
    }
}
best_predictors = (df_eval %>%
                       dplyr::filter(num_predictors == best_p))$predictors
best_predictors = strsplit(x = best_predictors,
                           split = ",")[[1]]
saveRDS(df_eval, "./data/df_eval_forward.rds")

# Plot:
make_subset_selection_plot(df_eval = df_eval,
                           df_plot = df_eval,
                           best_predictors = best_predictors)

# Estimated test MSE:
MSE_test_forward = (df_eval %>%
                       dplyr::filter(num_predictors == best_p))$cv_mse
MSE_test_forward

# Best model from Forward Stepwise Selection:
df_model = df_train_forward %>%
               dplyr::select(all_of(best_predictors),
                             all_of(response_var))
fit_forward = glm(formula = SalePrice ~ .,
              data = df_model)
multi_reg_plots(model_fit = fit_forward)

###### Ridge Regression





###### The Lasso




###### Principal Components Regression




###### Partial Least Squares




###### Comparison

# Plot:

# Best model:
# best_fit = fit_ ...



############ Prediction

y_pred = predict(best_fit,
                 newx = df_test_stand)
df_pred = data.frame(
    "Id" = df_test_stand$Id,
    "SalePrice" = y_pred
)
write.csv(object = df_pred,
          file = "./data/prediction.csv",
          sep = ",",
          row.names = FALSE)




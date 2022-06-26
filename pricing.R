
############################################ House Pricing Regression ############################################

require(ggplot2)
require(gridExtra)
require(grid)
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
source("./funcs/make_dim_reduc_plot.R")
source("./funcs/multi_reg_plots.R")
source("./funcs/make_comparison_plot.R")
source("./funcs/make_mest_models_plot.R")

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
df_test[response_var] = -1 # just to know that it is the test set
df_all = rbind(df_train,
               df_test)
df_all_orig = df_all

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
               dplyr::select(-all_of(response_var),
                             -all_of(id_var)) %>%
               dplyr::select_if(~!is.numeric(.)) %>%
               names()
num_vars = df_all %>%
               dplyr::select(-all_of(response_var),
                             -all_of(id_var)) %>%
               dplyr::select_if(~is.numeric(.)) %>%
               names()

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
df_all_cats_dumm = fastDummies::dummy_cols(.data = df_all %>%
                                                       dplyr::select(all_of(cat_vars)),
                                           select_columns = cat_vars,
                                           remove_selected_columns = TRUE,
                                           remove_first_dummy = TRUE)
dummy_vars = names(df_all_cats_dumm)

# Now it has a total of length(num_vars) + 1 response + 1 id + length(df_all_cats_dumm) variables:
df_all = cbind(df_all %>%
                   dplyr::select(-all_of(cat_vars)),
               df_all_cats_dumm)
p_predictors = df_all %>%
                   dplyr::select(-all_of(response_var),
                                 -all_of(id_var)) %>%
                   names()
predictors_not_dummies = num_vars

# Back to the train/test split:
df_train = df_all %>%
               dplyr::filter(eval(parse(text = response_var)) > -1)
df_test = df_all %>%
              dplyr::filter(eval(parse(text = response_var)) == -1) %>%
              dplyr::select(-all_of(response_var))

### Standardize the predictors from training and test

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

# Cross-validated MSE for a multiple linear regression:
cv_mse_mlr = function(predictors, df_model){
    # Fit:
    df_model = df_model[, c(response_var, predictors)]
    fit = glm(formula = paste(response_var, "~.",
                              collapse = ""),
              data = df_model)
    
    # Cross-validated MSE:
    cv_mse = boot::cv.glm(data = df_model,
                          glmfit = fit,
                          K = 10)$delta[1]
    return(cv_mse)
}

### K-Fold Cross-validated MSE

cv_mse = c()
cv_mse_se = c()
num_predictors = c()
predictors_names = c()
p = length(p_predictors)

# Null model:
fit = glm(formula = paste(response_var, "~1",
                          collapse = ""),
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
df_eval_forward = data.frame(
    "num_predictors" = num_predictors,
    "cv_mse" = cv_mse,
    "cv_mse_se" = cv_mse_se,
    "predictors" = predictors_names
)
df_eval_forward$cv_mse_se[is.na(df_eval_forward$cv_mse_se)] = 0
saveRDS(df_eval_forward, "./data/df_eval_forward.rds")
df_eval_forward = readRDS("./data/df_eval_forward.rds")

# Best model with the 1-standard-error rule:
min_cv_mse = min(df_eval_forward$cv_mse)
for(i in 2:nrow(df_eval_forward)){
    if(df_eval_forward$cv_mse[i] - df_eval_forward$cv_mse_se[i] <= min_cv_mse){
        best_p = i - 1
        break
    }
}
best_predictors = (df_eval_forward %>%
                       dplyr::filter(num_predictors == best_p))$predictors
best_predictors = strsplit(x = best_predictors,
                           split = ",")[[1]]

# Plot:
make_subset_selection_plot(df_eval = df_eval_forward,
                           df_plot = df_eval_forward,
                           best_predictors = best_predictors)

# Estimated test MSE:
test_mse_forward = (df_eval_forward %>%
                       dplyr::filter(num_predictors == best_p))$cv_mse
test_mse_se_forward = (df_eval_forward %>%
                          dplyr::filter(num_predictors == best_p))$cv_mse_se

# Best model from Forward Stepwise Selection:
df_model = df_train_forward %>%
               dplyr::select(all_of(best_predictors),
                             all_of(response_var))
fit_forward = glm(formula = paste(response_var, "~.",
                                  collapse = ""),
                  data = df_model)
saveRDS(fit_forward, "./data/fit_forward.rds")
fit_forward = readRDS("./data/fit_forward.rds")
multi_reg_plots(model_fit = fit_forward)

###### Ridge Regression

# Matrix data:
X = as.matrix(df_train_ridge[, p_predictors])
Y = df_train_ridge[, response_var]

# Fit:
fit_ridge = glmnet::glmnet(x = X,
                           y = Y,
                           alpha = 0,
                           standardize = FALSE,
                           family = "gaussian")

# K-Fold Cross-validation:
cv_ridge = glmnet::cv.glmnet(x = X,
                             y = Y,
                             alpha = 0,
                             type.measure = "mse",
                             nfolds = 10,
                             standardize = FALSE,
                             family = "gaussian")

# Plot:
make_shrinkage_plot(cv = cv_ridge,
                    model_type = "Ridge Regression",
                    fig_path = "./figs/Ridge_Regression.png")

# Estimated Test Prediction Error:
test_mse_ridge = cv_ridge$cvm[which(cv_ridge$lambda == cv_ridge$lambda.1se)]
test_mse_se_ridge = cv_ridge$cvsd[which(cv_ridge$lambda == cv_ridge$lambda.1se)]

###### The Lasso

# Matrix data:
X = as.matrix(df_train_lasso[, p_predictors])
Y = df_train_lasso[, response_var]

# Fit:
fit_lasso = glmnet::glmnet(x = X,
                           y = Y,
                           alpha = 1,
                           standardize = FALSE,
                           family = "gaussian")

# K-Fold Cross-validation:
cv_lasso = glmnet::cv.glmnet(x = X,
                             y = Y,
                             alpha = 1,
                             type.measure = "mse",
                             nfolds = 10,
                             standardize = FALSE,
                             family = "gaussian")

# Plot:
make_shrinkage_plot(cv = cv_lasso,
                    model_type = "The Lasso",
                    fig_path = "./figs/The_Lasso.png")

# Estimated Test Prediction Error:
test_mse_lasso = cv_lasso$cvm[which(cv_lasso$lambda == cv_lasso$lambda.1se)]
test_mse_se_lasso = cv_lasso$cvsd[which(cv_lasso$lambda == cv_lasso$lambda.1se)]

###### Principal Components Regression

# Fit:
fit_pcr = pls::pcr(formula = SalePrice ~ .,
                   data = df_train_pcr,
                   scale = FALSE,
                   validation = "CV",
                   segments = 10)

# K-Fold Cross-validation:
cv_pcr = pls::RMSEP(fit_pcr)
p = length(p_predictors)
m = 1:p
cv_pcr_MSE = (cv_pcr$val[seq(from = 3,
                             to = 2*p + 1,
                             by = 2)])**2
best_m = pls::selectNcomp(fit_pcr,
                          method = "onesigma",
                          plot = FALSE)
df_plot = data.frame(
    "m" = m,
    "cv_mse" = cv_pcr_MSE
)
df_best = data.frame(
    "best_m" = best_m,
    "cv_mse_best_m" = (df_plot %>%
                          dplyr::filter(m == best_m))$cv_mse[1]
)

# Plot:
make_dim_reduc_plot(df_plot = df_plot,
                    df_best = df_best)

# Estimated Test Prediction Error:
test_mse_pcr = df_best$cv_mse_best_m[1]
test_mse_pcr

###### Partial Least Squares

# Fit:
fit_pls = pls::plsr(formula = SalePrice ~ .,
                    data = df_train_pls,
                    scale = FALSE,
                    validation = "CV",
                    segments = 10)

# K-Fold Cross-validation:
cv_pls = pls::RMSEP(fit_pls)
p = length(p_predictors)
m = 1:p
cv_pls_MSE = (cv_pls$val[seq(from = 3,
                             to = 2*p + 1,
                             by = 2)])**2
best_m = pls::selectNcomp(fit_pls,
                          method = "onesigma",
                          plot = FALSE)
df_plot = data.frame(
    "m" = m,
    "cv_mse" = cv_pls_MSE
)
df_best = data.frame(
    "best_m" = best_m,
    "cv_mse_best_m" = (df_plot %>%
                          dplyr::filter(m == best_m))$cv_mse[1]
)

# Plot:
make_dim_reduc_plot(df_plot = df_plot %>%
                                  dplyr::filter(m < 200),
                    df_best = df_best)

# Estimated Test Prediction Error:
test_mse_pls = df_best$cv_mse_best_m[1]
test_mse_pls

###### Comparison

# Plot the estimated test Prediction Error by type of tuning parameter:
make_comparison_plot(cv_ridge = cv_ridge,
                     cv_lasso = cv_lasso,
                     fit_pcr = fit_pcr,
                     cv_pcr_MSE = cv_pcr_MSE,
                     fit_pls = fit_pls,
                     cv_pls_MSE = cv_pls_MSE,
                     df_eval_forward = df_eval_forward)

# Best models:
df_models_best = data.frame(
    "models" = c("Ridge", "Lasso", "Forward", "PCR", "PLS"),
    "cv_mse" = c(test_mse_ridge, test_mse_lasso, test_mse_forward, test_mse_pcr, test_mse_pls),
    "cv_mse_se" = c(test_mse_se_ridge, test_mse_se_lasso, test_mse_se_forward, NA, NA),
    stringsAsFactors = FALSE
)
make_mest_models_plot(df_models = df_models_best)

############ Prediction

### Estimated competition score for the test set

kaggle_score = function(y_pred, y_real, n_df, n_obs){
    estimated_score = sqrt(sum((log(y_pred) - log(y_real))**2)/(n_obs - n_df - 1))
    return(estimated_score)
}

# Training/Test split:
set.seed(666)
ind_test = sample(x = 1:nrow(df_train_stand),
                  size = trunc(0.4*nrow(df_train_stand)),
                  replace = FALSE)
df_train2 = df_train_stand[-ind_test, ]
df_test2 = df_train_stand[ind_test, ]
X_df_test2 = df_test2 %>%
                 dplyr::select(-all_of(response_var))

# Lasso:
y_pred_lasso = predict(fit_lasso,
                       as.matrix(X_df_test2),
                       s = cv_lasso$lambda.1se,
                       alpha = 1,
                       standardize = FALSE,
                       family = "gaussian") %>%
                   as.numeric()
estimated_score = kaggle_score(y_pred = y_pred_lasso[y_pred_lasso > 0],
                               y_real = df_test2[response_var],
                               n_df = fit_lasso$df[which(cv_lasso$lambda == cv_lasso$lambda.1se)],
                               n_obs = nrow(df_test2))
estimated_score

# Ridge:
y_pred_ridge = predict(fit_ridge,
                       as.matrix(X_df_test2),
                       s = cv_ridge$lambda.1se,
                       alpha = 1,
                       standardize = FALSE,
                       family = "gaussian") %>%
                   as.numeric()
estimated_score = kaggle_score(y_pred = y_pred_ridge[y_pred_ridge > 0],
                               y_real = df_test2[response_var],
                               n_df = fit_ridge$df[which(cv_ridge$lambda == cv_ridge$lambda.1se)],
                               n_obs = nrow(df_test2))
estimated_score

# Forward:
y_pred_forward = predict(fit_forward,
                         X_df_test2) %>%
                     as.numeric()
estimated_score = kaggle_score(y_pred = y_pred_forward[y_pred_forward > 0],
                               y_real = df_test2[response_var],
                               n_df = length(fit_forward$coefficients) - 1,
                               n_obs = nrow(df_test2))
estimated_score

# PCR:
y_pred_pcr = predict(fit_pcr,
                     newdata = X_df_test2,
                     ncomp = pls::selectNcomp(fit_pcr,
                                           method = "onesigma",
                                           plot = FALSE)) %>%
                 as.numeric()
estimated_score = kaggle_score(y_pred = y_pred_pcr[y_pred_pcr > 0],
                               y_real = df_test2[response_var],
                               n_df = pls::selectNcomp(fit_pcr,
                                                       method = "onesigma",
                                                       plot = FALSE),
                               n_obs = nrow(df_test2))
estimated_score

# PLS:
y_pred_pls = predict(fit_pls,
                     newdata = X_df_test2,
                     ncomp = pls::selectNcomp(fit_pls,
                                           method = "onesigma",
                                           plot = FALSE)) %>%
                 as.numeric()
estimated_score = kaggle_score(y_pred = y_pred_pls[y_pred_pls > 0],
                               y_real = df_test2[response_var],
                               n_df = pls::selectNcomp(fit_pls,
                                                       method = "onesigma",
                                                       plot = FALSE),
                               n_obs = nrow(df_test2))
estimated_score

# Ensemble model:
y_pred = data.frame(
    y_pred_lasso,
    y_pred_ridge,
    y_pred_forward,
    y_pred_pcr,
    y_pred_pls
) %>%
    rowMeans()
estimated_score = kaggle_score(y_pred = y_pred_pls[y_pred_pls > 0],
                               y_real = df_test2[response_var],
                               n_df = pls::selectNcomp(fit_pls,
                                                       method = "onesigma",
                                                       plot = FALSE),
                               n_obs = nrow(df_test2))
estimated_score

### Predict with an ensemble model

df_test_final = df_test_stand %>%
                     dplyr::select(-all_of(id_var))
y_pred_lasso = predict(fit_lasso,
                       as.matrix(df_test_final),
                       s = cv_lasso$lambda.1se,
                       alpha = 1,
                       standardize = FALSE,
                       family = "gaussian") %>%
                   as.numeric()
y_pred_ridge = predict(fit_ridge,
                       as.matrix(df_test_final),
                       s = cv_ridge$lambda.1se,
                       alpha = 1,
                       standardize = FALSE,
                       family = "gaussian") %>%
                   as.numeric()
y_pred_forward = predict(fit_forward,
                         df_test_final) %>%
                     as.numeric()
y_pred_pcr = predict(fit_pcr,
                     newdata = df_test_final,
                     ncomp = pls::selectNcomp(fit_pcr,
                                           method = "onesigma",
                                           plot = FALSE)) %>%
                 as.numeric()
y_pred_pls = predict(fit_pls,
                     newdata = df_test_final,
                     ncomp = pls::selectNcomp(fit_pls,
                                           method = "onesigma",
                                           plot = FALSE)) %>%
                 as.numeric()
y_pred = data.frame(
    y_pred_lasso,
    y_pred_ridge,
    y_pred_forward,
    y_pred_pcr,
    y_pred_pls
) %>%
    rowMeans()
y_pred[y_pred < 0] = 0
df_pred = data.frame(
    "Id" = df_test_stand$Id,
    "SalePrice" = y_pred
)
write.csv(df_pred,
          file = "./data/submission_ensemble.csv",
          row.names = FALSE)



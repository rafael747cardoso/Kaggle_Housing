
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
saveRDS(df_eval, "./data/df_eval_forward.rds")

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

# Plot:
make_subset_selection_plot(df_eval = df_eval,
                           df_plot = df_eval,
                           best_predictors = best_predictors)

# Estimated test MSE:
test_mse_forward = (df_eval %>%
                       dplyr::filter(num_predictors == best_p))$cv_mse
test_mse_forward

# Best model from Forward Stepwise Selection:
df_model = df_train_forward %>%
               dplyr::select(all_of(best_predictors),
                             all_of(response_var))
fit_forward = glm(formula = SalePrice ~ .,
              data = df_model)
saveRDS(fit_forward, "./data/fit_forward.rds")
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
test_mse_ridge

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
test_mse_lasso

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
                          dplyr::filter(m == best_m))$cv_pls[1]
)

# Plot:
make_dim_reduc_plot(df_plot = df_plot,
                    df_best = df_best)

# Estimated Test Prediction Error:
test_mse_pls = df_best$cv_mse_best_m[1]
test_mse_pls

###### Comparison

### Plot the estimated test Prediction Error by type of tuning parameter

## by lambda (Ridge and Lasso)

# Ridge:
best_lambda_ridge = cv_ridge$lambda.1se
df_mse_ridge = data.frame(
    "cv_mse" = cv_ridge$cvm,
    "cv_mse_se" = cv_ridge$cvsd,
    "lambdas" = cv_ridge$lambda,
    "d" = fit_ridge$df
)

# Lasso:
best_lambda_lasso = cv_lasso$lambda.1se
df_mse_lasso = data.frame(
    "cv_mse" = cv_lasso$cvm,
    "cv_mse_se" = cv_lasso$cvsd,
    "lambdas" = cv_lasso$lambda,
    "d" = fit_lasso$df
)

color_ridge = "#f68105"
color_lasso = "#0591f6"
p1 = ggplot() +
    # Ridge:
    geom_point(
        data = df_mse_ridge,
        aes(
            x = log(lambdas),
            y = cv_mse,
            color = "Ridge"
        ),
        size = 2
    ) +
    geom_errorbar(
        data = df_mse_ridge,
        aes(
            x = log(lambdas),
            y = cv_mse,
            ymin = cv_mse - cv_mse_se,
            ymax = cv_mse + cv_mse_se
        ),
        color = color_ridge,
        width = 0.1
    ) +
    geom_vline(
        aes(
            xintercept = log(cv_ridge$lambda.min),
            color = "Ridge Minimum Lambda"
        ),
        size = 1,
        linetype = "dashed",
        show.legend = TRUE
    ) +
    geom_vline(
        aes(
            xintercept = log(best_lambda_ridge),
            color = "Ridge Largest Lambda within\n1SE of the minimum"
        ),
        size = 1,
        linetype = "dashed",
        show.legend = TRUE
    ) +
    geom_hline(
        yintercept = cv_ridge$cvm[which(cv_ridge$lambda == best_lambda_ridge)] - 
                     cv_ridge$cvsd[which(cv_ridge$lambda == best_lambda_ridge)],
        color = "grey",
        linetype = "dashed"
    ) +
    # Lasso:
    geom_point(
        data = df_mse_lasso,
        aes(
            x = log(lambdas),
            y = cv_mse,
            color = "Lasso"
        ),
        size = 2
    ) +
    geom_errorbar(
        data = df_mse_lasso,
        aes(
            x = log(lambdas),
            y = cv_mse,
            ymin = cv_mse - cv_mse_se,
            ymax = cv_mse + cv_mse_se
        ),
        color = color_lasso,
        width = 0.1
    ) +
    geom_vline(
        aes(
            xintercept = log(cv_lasso$lambda.min),
            color = "Lasso Minimum Lambda"
        ),
        size = 1,
        linetype = "dashed",
        show.legend = TRUE
    ) +
    geom_vline(
        aes(
            xintercept = log(best_lambda_lasso),
            color = "Lasso Largest Lambda within\n1SE of the minimum"
        ),
        size = 1,
        linetype = "dashed",
        show.legend = TRUE
    ) +
    geom_hline(
        yintercept = cv_lasso$cvm[which(cv_lasso$lambda == best_lambda_lasso)] - 
                     cv_lasso$cvsd[which(cv_lasso$lambda == best_lambda_lasso)],
        color = "grey",
        linetype = "dashed"
    ) +
    scale_colour_manual(
        values = c("Ridge" = color_ridge,
                   "Ridge Minimum Lambda" = "#d1976b",
                   "Ridge Largest Lambda within\n1SE of the minimum" = "#b75309",
                   "Lasso" = color_lasso,
                   "Lasso Minimum Lambda" = "#6a91cc",
                   "Lasso Largest Lambda within\n1SE of the minimum" = "#0e469a"),
        guide = guide_legend(ncol = 2)
    ) +
    theme(
        axis.text.x = element_text(
            size = 14,
            angle = 0,
            hjust = 0.5,
            vjust = 1
        ),
        axis.text.y = element_text(
            size = 14
        ),
        axis.title.x = element_text(
            size = 15,
            face = "bold"
        ),
        axis.title.y = element_text(
            size = 15,
            face = "bold"
        ),
        panel.background = element_rect(
            fill = "white"
        ),
        panel.grid.major = element_line(
            size = 0.2,
            linetype = "solid",
            colour = "#eaeaea"
        ),
        panel.grid.minor = element_line(
            size = 0.1,
            linetype = "solid",
            colour = "#eaeaea"
        ),
        legend.title = element_blank(),
        legend.text = element_text(
            size = 12
        ),
        legend.position = "top",
        legend.background = element_rect(
            fill = "transparent"
        )
    ) +
    xlab("Log(lambda)") +
    ylab(cv_ridge$name)

## by number of components or predictors (PCR, PLS and Forward Selection)

# PCR:
p = length(p_predictors)
best_m_pcr = pls::selectNcomp(fit_pcr,
                              method = "onesigma",
                              plot = FALSE)
df_mse_pcr = data.frame(
    "m" = 1:p,
    "cv_mse" = cv_pcr_MSE
)

# PLS:
best_m_pls = pls::selectNcomp(fit_pls,
                              method = "onesigma",
                              plot = FALSE)
df_mse_pls = data.frame(
    "m" = 1:p,
    "cv_mse" = cv_pls_MSE
)

# Forward Selection:
df_mse_forward = data.frame(
    "k" = num_predictors,
    "cv_mse" = cv_mse,
    "cv_mse_se" = cv_mse_se
)
df_mse_forward$cv_mse_se[is.na(df_mse_forward$cv_mse_se)] = 0
min_cv_mse = min(df_mse_forward$cv_mse)
for(i in 2:nrow(df_mse_forward)){
    if(df_mse_forward$cv_mse[i] - df_mse_forward$cv_mse_se[i] <= min_cv_mse){
        best_k_forward = i - 1
        break
    }
}

color_pcr = "#43c41a"
color_pls = "#8621c4"
color_forward = "#df3f32"
p2 = ggplot() +
    # PCR:
    geom_point(
        data = df_mse_pcr,
        aes(
            x = m,
            y = cv_mse,
            color = "PCR"
        ),
        size = 2
    ) +
    geom_vline(
        aes(
            xintercept = best_m_pcr,
            color = "PCR Largest m within\n1SE of the minimum"
        ),
        size = 1,
        linetype = "dashed",
        show.legend = TRUE
    ) +
    geom_hline(
        yintercept = df_mse_pcr$cv_mse[which(df_mse_pcr$m == best_m_pcr)],
        color = "grey",
        linetype = "dashed"
    ) +
    # PLS:
    geom_point(
        data = df_mse_pls,
        aes(
            x = m,
            y = cv_mse,
            color = "PLS"
        ),
        size = 2
    ) +
    geom_vline(
        aes(
            xintercept = best_m_pls,
            color = "PLS Largest m within\n1SE of the minimum"
        ),
        size = 1,
        linetype = "dashed",
        show.legend = TRUE
    ) +
    geom_hline(
        yintercept = df_mse_pls$cv_mse[which(df_mse_pls$m == best_m_pls)],
        color = "grey",
        linetype = "dashed"
    ) +
    # Forward Selection:
    geom_point(
        data = df_mse_forward,
        aes(
            x = k,
            y = cv_mse,
            color = "Forward Selection"
        ),
        size = 2
    ) +
    geom_vline(
        aes(
            xintercept = best_k_forward,
            color = "Forward Selection Largest k within\n1SE of the minimum"
        ),
        size = 1,
        linetype = "dashed",
        show.legend = TRUE
    ) +
    geom_hline(
        yintercept = df_mse_forward$cv_mse[which(df_mse_forward$k == best_k_forward)],
        color = "grey",
        linetype = "dashed"
    ) +
    scale_colour_manual(
        values = c("PCR" = color_pcr,
                   "PCR Largest m within\n1SE of the minimum" = "#2f8d11",
                   "PLS" = color_pls,
                   "PLS Largest m within\n1SE of the minimum" = "#722c80",
                   "Forward Selection" = color_forward,
                   "Forward Selection Largest m within\n1SE of the minimum" = "#8a2c24"),
        guide = guide_legend(ncol = 3)
    ) +
    theme(
        axis.text.x = element_text(
            size = 14,
            angle = 0,
            hjust = 0.5,
            vjust = 1
        ),
        axis.text.y = element_text(
            size = 14
        ),
        axis.title.x = element_text(
            size = 15,
            face = "bold"
        ),
        axis.title.y = element_text(
            size = 15,
            face = "bold"
        ),
        panel.background = element_rect(
            fill = "white"
        ),
        panel.grid.major = element_line(
            size = 0.2,
            linetype = "solid",
            colour = "#eaeaea"
        ),
        panel.grid.minor = element_line(
            size = 0.1,
            linetype = "solid",
            colour = "#eaeaea"
        ),
        legend.title = element_blank(),
        legend.text = element_text(
            size = 12
        ),
        legend.position = "top",
        legend.background = element_rect(
            fill = "transparent"
        )
    ) +
    xlab("Number of components or predictors") +
    ylab("CV MSE")

grid.arrange(p1, p2,
             nrow = 1,
             ncol = 2,
             top = textGrob("Model Selection Comparison",
                            gp = gpar(
                                fontsize = 16,
                                font = 3
                            )
                        )
            )

# Best model:
df_models = data.frame(
    "models" = c("Ridge", "Lasso", "Forward", "PCR", "PLS"),
    "cv_mse" = c(test_mse_ridge, test_mse_lasso, test_mse_forward, test_mse_pcr, test_mse_pls),
    stringsAsFactors = FALSE
)
best_model_type = (df_models %>%
                      dplyr::filter(cv_mse == min(cv_mse)))$models[1] %>%
                      tolower()
best_fit = eval(parse(text = paste0("fit_", best_model_type)))

############ Prediction

y_pred = predict(best_fit,
                 newx = df_test_stand)
df_pred = data.frame(
    "Id" = df_test_stand$Id,
    "SalePrice" = y_pred
)
write.csv(object = df_pred,
          file = "./data/prediction_submission_version_1.csv",
          sep = ",",
          row.names = FALSE)




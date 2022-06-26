
### Comparison of the estimated test Prediction Error by type of tuning parameter

make_comparison_plot = function(cv_ridge, cv_lasso, fit_pcr, cv_pcr_MSE, fit_pls, cv_pls_MSE, df_eval_forward){
    ## by lambda (Ridge and Lasso)
    
    # Ridge:
    best_lambda_ridge = cv_ridge$lambda.1se
    df_mse_ridge = data.frame(
        "cv_mse" = cv_ridge$cvm,
        "cv_mse_se" = cv_ridge$cvsd,
        "lambdas" = cv_ridge$lambda,
        "d" = cv_ridge$glmnet.fit$df
    )
    
    # Lasso:
    best_lambda_lasso = cv_lasso$lambda.1se
    df_mse_lasso = data.frame(
        "cv_mse" = cv_lasso$cvm,
        "cv_mse_se" = cv_lasso$cvsd,
        "lambdas" = cv_lasso$lambda,
        "d" = cv_lasso$glmnet.fit$df
    )
    
    y_greatest_lass_ridg = 1.1*max(c(max(df_mse_ridge$cv_mse),
                                     max(df_mse_lasso$cv_mse)))
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
        geom_line(
            data = df_mse_ridge,
            aes(
                x = log(lambdas),
                y = cv_mse,
                color = "Ridge"
            ),
            size = 1
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
        geom_line(
            data = df_mse_lasso,
            aes(
                x = log(lambdas),
                y = cv_mse,
                color = "Lasso"
            ),
            size = 1
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
        ylim(0, y_greatest_lass_ridg) +
        xlim(max(c(max(log(df_mse_ridge$lambdas)), max(log(df_mse_lasso$lambdas)))),
             min(c(min(log(df_mse_ridge$lambdas)), min(log(df_mse_lasso$lambdas))))) +
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
        "k" = df_eval_forward$num_predictors,
        "cv_mse" = df_eval_forward$cv_mse,
        "cv_mse_se" = df_eval_forward$cv_mse_se
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
        geom_line(
            data = df_mse_pcr,
            aes(
                x = m,
                y = cv_mse,
                color = "PCR"
            ),
            size = 1
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
        geom_line(
            data = df_mse_pls,
            aes(
                x = m,
                y = cv_mse,
                color = "PLS"
            ),
            size = 1
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
        geom_line(
            data = df_mse_forward,
            aes(
                x = k,
                y = cv_mse,
                color = "Forward Selection"
            ),
            size = 1
        ) +
        geom_errorbar(
            data = df_mse_forward,
            aes(
                x = k,
                y = cv_mse,
                ymin = cv_mse - cv_mse_se,
                ymax = cv_mse + cv_mse_se
            ),
            color = color_forward,
            width = 0.1
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
                       "Forward Selection Largest k within\n1SE of the minimum" = "#8a2c24"),
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
        ylim(0, y_greatest_lass_ridg) +
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

}


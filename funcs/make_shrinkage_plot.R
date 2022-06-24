
### Regularization plots

make_shrinkage_plot = function(cv, model_type, fig_path){

    require(RColorBrewer)
    require(grid)
    require(gridExtra)
    
    ### Deviance versus lambda
    
    model_fit = cv$glmnet.fit
    best_lambda = cv$lambda.1se
    
    # Estimated Test Deviance:
    df_deviance = data.frame(
        "cv_deviance" = cv$cvm,
        "cv_deviance_se" = cv$cvsd,
        "lambdas" = cv$lambda,
        "d" = model_fit$df
    )
    
    # Shrinkage amount:
    p_original = model_fit$dim[1]
    p_best = as.numeric(cv$nzero[cv$lambda == best_lambda])
    
    # Plot:
    p1 = ggplot() +
    geom_point(
        data = df_deviance,
        aes(
            x = log(lambdas),
            y = cv_deviance
        ),
        color = "red",
        size = 2
    ) +
    geom_errorbar(
        data = df_deviance,
        aes(
            x = log(lambdas),
            y = cv_deviance,
            ymin = cv_deviance - cv_deviance_se,
            ymax = cv_deviance + cv_deviance_se
        ),
        width = 0.1
    ) +
    geom_vline(
        aes(
            xintercept = log(cv$lambda.min),
            color = "Minimum Lambda"
        ),
        size = 1,
        linetype = "dashed",
        show.legend = TRUE
    ) +
    geom_vline(
        aes(
            xintercept = log(best_lambda),
            color = "Largest Lambda within\n1SE of the minimum"
        ),
        size = 1,
        linetype = "dashed",
        show.legend = TRUE
    ) +
    geom_hline(
        yintercept = cv$cvm[which(cv$lambda == best_lambda)] - 
                     cv$cvsd[which(cv$lambda == best_lambda)],
        color = "grey",
        linetype = "dashed"
    ) +
    scale_colour_manual(
        values = c("Minimum Lambda" = "orange",
                   "Largest Lambda within\n1SE of the minimum" = "blue")
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
    ylab(cv$name)
    
    ### Coefficients versus lambda
    
    # Coefficient values for each lambda:
    df_coefs = NULL
    for(i in 2:model_fit$dim[1]){
        df_coefs = rbind(df_coefs,
            data.frame(
                "d" = model_fit$df,
                "lambda" = model_fit$lambda,
                "dev" = model_fit$dev.ratio,
                "coef_name" = row.names(coef(model_fit))[i],
                "coef_val" = as.numeric(coef(model_fit)[i, ])
            )
        )
    }
    l1norm = c()
    for(i in 1:model_fit$dim[2]){
        l1norm = c(l1norm,
                   sum(abs(coef(model_fit,
                        s = model_fit$lambda[i])[-1]))
                   )
    }
    df_coefs$l1norm = l1norm

    # Color only the most significant predictors at lambda = best_lambda:
    df_sorted = df_coefs %>%
                    dplyr::filter(lambda == best_lambda) %>%
                    dplyr::arrange(desc(abs(coef_val)))
    n_best = ifelse(model_fit$dim[1] > 10,
                    10,
                    model_fit$dim[1] %/% 2)
    predictors_best = df_sorted$coef_name[1:n_best]
    predictors_others = df_sorted$coef_name[(n_best + 1):(model_fit$dim[1] - 1)]
    df_best = df_coefs %>% 
                 dplyr::filter(coef_name %in% predictors_best)
    df_others = df_coefs %>% 
                 dplyr::filter(coef_name %in% predictors_others)
    color_best_1 = "#44C800"
    color_best_2 = "#0031C8"
    color_others = "#BABABA"
    my_palette = colorRampPalette(c(color_best_1, color_best_2,
                                    rep(color_others, trunc(model_fit$dim[1]/n_best))))
    
    # Plot:
    p2 = ggplot() +
    geom_line(
        data = df_others,
        aes(
            x = log(lambda),
            y = coef_val,
            color = coef_name
        ),
        size = 1,
        show.legend = FALSE
    )+
    geom_line(
        data = df_best,
        aes(
            x = log(lambda),
            y = coef_val,
            color = coef_name
        ),
        size = 1,
        show.legend = TRUE
    ) +
    scale_color_manual(
        values = my_palette(length(unique(df_coefs$coef_name))),
        name = paste0("Original p = ", p_original,
                      "\nBest p       = ", p_best,
                      "\n\n\nTop predictors at best lambda"),
        breaks = predictors_best
    ) +
    geom_vline(
        xintercept = log(cv$lambda.min),
        color = "orange",
        size = 1,
        linetype = "dashed",
        show.legend = FALSE
    ) +
    geom_vline(
        xintercept = log(best_lambda),
        color = "blue",
        size = 1,
        linetype = "dashed",
        show.legend = FALSE
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
        legend.title = element_text(
            face = "bold",
            size = 14
        ),
        legend.text = element_text(
            size = 12
        ),
        legend.position = "right",
        legend.background = element_rect(
            fill = "transparent"
        )
    ) +
    xlab("Log(lambda)") +
    ylab("Coefficients")
    
    ### Coefficients versus l1-norm
    
    # Plot:
    p3 = ggplot() +
    geom_line(
        data = df_others %>%
                   dplyr::arrange(l1norm),
        aes(
            x = l1norm,
            y = coef_val,
            color = coef_name
        ),
        size = 1,
        show.legend = FALSE
    ) +
    geom_line(
        data = df_best %>%
                   dplyr::arrange(l1norm),
        aes(
            x = l1norm,
            y = coef_val,
            color = coef_name
        ),
        size = 1,
        show.legend = FALSE
    ) +
    scale_color_manual(
        values = my_palette(length(unique(df_coefs$coef_name))),
        name = "Top predictors at best lambda",
        breaks = predictors_best
    ) +
    geom_vline(
        xintercept = (df_coefs %>%
                         dplyr::filter(lambda == cv$lambda.min))$l1norm[1],
        color = "orange",
        size = 1,
        linetype = "dashed",
        show.legend = FALSE
    ) +
    geom_vline(
        xintercept = (df_coefs %>%
                         dplyr::filter(lambda == best_lambda))$l1norm[1],
        color = "blue",
        size = 1,
        linetype = "dashed",
        show.legend = FALSE
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
        legend.text = element_text(
            size = 12
        ),
        legend.position = "right",
        legend.background = element_rect(
            fill = "transparent"
        )
    ) +
    xlab("L1-Norm") +
    ylab("Coefficients")
    
    ### Coefficients versus deviance
    
    # Plot:
    p4 = ggplot() +
    geom_line(
        data = df_others,
        aes(
            x = dev,
            y = coef_val,
            color = coef_name
        ),
        size = 1,
        show.legend = FALSE
    )+
    geom_line(
        data = df_best,
        aes(
            x = dev,
            y = coef_val,
            color = coef_name
        ),
        size = 1,
        show.legend = FALSE
    ) +
    scale_color_manual(
        values = my_palette(length(unique(df_coefs$coef_name))),
        name = "Top predictors at best lambda",
        breaks = predictors_best
    ) +
    geom_vline(
        xintercept = (df_coefs %>%
                         dplyr::filter(lambda == cv$lambda.min))$dev[1],
        color = "orange",
        size = 1,
        linetype = "dashed",
        show.legend = FALSE
    ) +
    geom_vline(
        xintercept = (df_coefs %>%
                         dplyr::filter(lambda == best_lambda))$dev[1],
        color = "blue",
        size = 1,
        linetype = "dashed",
        show.legend = FALSE
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
        legend.text = element_text(
            size = 12
        ),
        legend.position = "right",
        legend.background = element_rect(
            fill = "transparent"
        )
    ) +
    xlab("Fraction Deviance Explained") +
    ylab("Coefficients")
    
    # Assemble the plots:
    ggsave(fig_path,
           plot = grid.arrange(p1, p2, p3, p4,
                 nrow = 2,
                 ncol = 2,
                 top = textGrob(model_type,
                                gp = gpar(
                                    fontsize = 16,
                                    font = 3
                                )
                            )
                ),
           device = "png",
           scale = 2.7,
           width = 1900,
           height = 1080,
           units = "px")
}


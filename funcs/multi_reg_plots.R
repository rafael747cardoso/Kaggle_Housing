
### Multiple regression plots

multi_reg_plots = function(model_fit){
    
    require(gridExtra)
    
    fit_trend = function(y_var, x_var, df){
        names(df)[which(names(df) == y_var)] = "y_var"
        names(df)[which(names(df) == x_var)] = "x_var"
        lo = loess(formula = y_var ~ x_var,
                   data = df,
                   span = 1)
        return(predict(lo))
    }
    
    # Residuals versus Predicted values:
    df_res = data.frame(
        "y_pred" = model_fit$fitted.values,
        "y_residual" = model_fit$residuals
    )
    df_res$trend_res = fit_trend(y_var = "y_residual",
                                 x_var = "y_pred",
                                 df = df_res)
    p1 = ggplot(
        data = df_res
    ) +
    geom_point(
        aes(
            x = y_pred,
            y = y_residual
        ),
        size = 1,
        color = "#5580f1",
        alpha = 0.4
    ) +
    geom_hline(
        yintercept = 0,
        color = "#b0e18d",
        linetype = "dashed"
    ) +
    geom_line(
        aes(
            x = y_pred,
            y = trend_res
        ),
        color = "#d52c36",
        size = 1,
        alpha = 0.6
    ) +
    theme(
        axis.text.x = element_text(
            size = 12,
            hjust = 0.5,
            vjust = 1
        ),
        axis.title.x = element_text(
            size = 13,
            face = "bold"
        ),
        axis.text.y = element_text(
            size = 12
        ),
        axis.title.y = element_text(
            size = 13,
            face = "bold"
        ),
        panel.background = element_rect(
            fill = "white"
        ),
        panel.grid.major = element_line(
            size = 0.2,
            linetype = "solid",
            colour = "#d1caca"
        ),
        panel.grid.minor = element_line(
            size = 0.1,
            linetype = "solid",
            colour = "#d1caca"
        ),
        plot.margin = margin(
            t = 20, 
            r = 5,
            b = 5, 
            l = 5,
            unit = "pt"
        ),
        plot.title = element_text(
            size = 14,
            face = "bold",
            hjust = 0.5
        )
    ) +
    ggtitle("Residual versus Predicted values") +
    xlab("Predicted values") +
    ylab("Residual")
    
    # Normal Q-Q: std. Pearson residual versus theoretical quantiles
    df_res$std_residual = df_res$y_residual/sd(df_res$y_residual)
    df_res_sorted = df_res %>%
                        dplyr::arrange(df_res$y_residual)
    df_res_sorted$norm_quant = quantile(rnorm(n = nrow(df_res_sorted),
                                              mean = 0,
                                              sd = 1),
                                        probs = seq(from = 0.001,
                                                    to = 0.999,
                                                    length.out = nrow(df_res_sorted)))
    p2 = ggplot(
        data = df_res_sorted
    ) +
    geom_point(
        aes(
            x = norm_quant,
            y = std_residual
        ),
        size = 1,
        color = "#5580f1",
        alpha = 0.4
    ) +
    geom_line(
        aes(
            x = norm_quant,
            y = norm_quant
        ),
        color = "#438712",
        linetype = "dashed",
        size = 1,
        alpha = 0.6
    ) +
    theme(
        axis.text.x = element_text(
            size = 12,
            hjust = 0.5,
            vjust = 1
        ),
        axis.title.x = element_text(
            size = 13,
            face = "bold"
        ),
        axis.text.y = element_text(
            size = 12
        ),
        axis.title.y = element_text(
            size = 13,
            face = "bold"
        ),
        panel.background = element_rect(
            fill = "white"
        ),
        panel.grid.major = element_line(
            size = 0.2,
            linetype = "solid",
            colour = "#d1caca"
        ),
        panel.grid.minor = element_line(
            size = 0.1,
            linetype = "solid",
            colour = "#d1caca"
        ),
        plot.margin = margin(
            t = 20, 
            r = 5,
            b = 5, 
            l = 5,
            unit = "pt"
        ),
        plot.title = element_text(
            size = 14,
            face = "bold",
            hjust = 0.5
        )
    ) +
    ggtitle("Normal Q-Q") +
    xlab("Theoretical quantiles") +
    ylab("Standardized Pearson Residual")

    # Scale-location: sqrt(std. Pearson residual) versus predicted values
    df_res$sqrt_std_residual = sqrt(abs(df_res$std_residual))
    df_res$trend_sqrt_std_res = fit_trend(y_var = "sqrt_std_residual",
                                          x_var = "y_pred",
                                          df = df_res)
    p3 = ggplot(
        data = df_res
    ) +
    geom_point(
        aes(
            x = y_pred,
            y = sqrt_std_residual
        ),
        size = 1,
        color = "#5580f1",
        alpha = 0.4
    ) +
    geom_line(
        aes(
            x = y_pred,
            y = trend_sqrt_std_res
        ),
        color = "#d52c36",
        size = 1,
        alpha = 0.6
    ) +
    theme(
        axis.text.x = element_text(
            size = 12,
            hjust = 0.5,
            vjust = 1
        ),
        axis.title.x = element_text(
            size = 13,
            face = "bold"
        ),
        axis.text.y = element_text(
            size = 12
        ),
        axis.title.y = element_text(
            size = 13,
            face = "bold"
        ),
        panel.background = element_rect(
            fill = "white"
        ),
        panel.grid.major = element_line(
            size = 0.2,
            linetype = "solid",
            colour = "#d1caca"
        ),
        panel.grid.minor = element_line(
            size = 0.1,
            linetype = "solid",
            colour = "#d1caca"
        ),
        plot.margin = margin(
            t = 20, 
            r = 5,
            b = 5, 
            l = 5,
            unit = "pt"
        ),
        plot.title = element_text(
            size = 14,
            face = "bold",
            hjust = 0.5
        )
    ) +
    ggtitle("Scale-Location") +
    xlab("Predicted values") +
    ylab("sqrt(|Standardized Pearson Residual|)")
    
    # Residuals versus Leverage: std. Pearson residual versus Leverage
    df_res$leverage = as.numeric(hatvalues(model_fit))
    df_res$trend_res_lev = fit_trend(y_var = "std_residual",
                                     x_var = "leverage",
                                     df = df_res)
    df_res$cook_dist = cooks.distance(model_fit)

    cook_dist_contour = function(leverage, level, model, side_positive){
        return(side_positive*sqrt(level*length(coef(model))*(1 - leverage)/leverage))
    }
    
    p4 = ggplot() +
    geom_point(
        data = df_res,
        aes(
            x = leverage,
            y = std_residual
        ),
        color = "#5580f1",
        size = 1,
        alpha = 0.4
    ) +
    stat_function(
        fun = cook_dist_contour,
        args = list(
            level = 0.5,
            model = model_fit,
            side_positive = 1
        ),
        linetype = "dashed",
        aes(
            colour = "Cook's Distance = 0.5",
        )
    ) +
    stat_function(
        fun = cook_dist_contour,
        args = list(
            level = 0.5,
            model = model_fit,
            side_positive = -1
        ),
        linetype = "dashed",
        aes(
            colour = "Cook's Distance = 0.5",
        )
    ) +
    scale_colour_manual(
        values = c("Cook's Distance = 0.5" = "#9f71b7")
    ) +
    geom_line(
        data = df_res,
        aes(
            x = leverage,
            y = trend_res_lev
        ),
        color = "#d52c36",
        size = 1,
        alpha = 0.6
    ) +
    geom_hline(
        yintercept = 0,
        color = "#b0e18d",
        linetype = "dashed"
    ) +
    theme(
        axis.text.x = element_text(
            size = 12,
            hjust = 0.5,
            vjust = 1
        ),
        axis.title.x = element_text(
            size = 13,
            face = "bold"
        ),
        axis.text.y = element_text(
            size = 12
        ),
        axis.title.y = element_text(
            size = 13,
            face = "bold"
        ),
        panel.background = element_rect(
            fill = "white"
        ),
        panel.grid.major = element_line(
            size = 0.2,
            linetype = "solid",
            colour = "#d1caca"
        ),
        panel.grid.minor = element_line(
            size = 0.1,
            linetype = "solid",
            colour = "#d1caca"
        ),
        plot.margin = margin(
            t = 20, 
            r = 5,
            b = 5, 
            l = 5,
            unit = "pt"
        ),
        plot.title = element_text(
            size = 14,
            face = "bold",
            hjust = 0.5
        ),
        legend.title = element_blank(),
        legend.text = element_text(
            size = 14
        ),
        legend.position = c(0.3, 0.1),
        legend.background = element_rect(
            fill = "transparent"
        )
    ) +
    coord_cartesian(
        xlim = c(min(df_res$leverage), 
                 1.2*max(df_res$leverage)),
        ylim = c(sign(min(df_res$std_residual))*1.2*abs(min(df_res$std_residual)), 
                 1.2*max(df_res$std_residual))
    ) +
    scale_x_continuous(
        limits = c(min(df_res$leverage), 
                   1.2*max(df_res$leverage))
    ) +
    ggtitle("Residual versus Leverage") +
    xlab("Leverage") +
    ylab("Standardized Pearson Residual")
    
    # Assemble the plots:
    grid.arrange(p1, p2, p3, p4,
                 nrow = 2,
                 ncol = 2)
}


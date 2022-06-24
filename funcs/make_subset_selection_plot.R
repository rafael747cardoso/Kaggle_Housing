
### Plot the Cross-validated MSE and the Adjusted R² versus the number of predictors

get_integer_breaks = function(input_vector, interval = 1){
    minimum = floor(min(input_vector))
    maximum = ceiling(max(input_vector))
    breaks = seq(from = minimum,
                 to = maximum,
                 by = interval)
    return(breaks)
}

make_subset_selection_plot = function(df_eval, df_plot, best_predictors){

    y_var_name = names(df_eval)[2]
    names(df_eval)[1:3] = c("num_predictors", "cv_deviance", "cv_deviance_se")
    names(df_plot)[1:3] = c("num_predictors", "cv_deviance", "cv_deviance_se")
    
    # Best models:
    df_best = df_eval %>%
                  dplyr::filter(predictors == paste(best_predictors, collapse = ','))
    
    # Discart the null model:
    df_plot = df_plot[2:nrow(df_plot), ]
    df_eval = df_eval[2:nrow(df_eval), ]
    
    # CV MSE:
    ggplot() +
    geom_point(
        data = df_eval,
        aes(
            x = num_predictors,
            y = cv_deviance
        ),
        color = "gray",
        size = 1,
        alpha = 0.7
    ) +
    geom_errorbar(
        data = df_plot,
        aes(
            x = num_predictors,
            y = cv_deviance,
            ymin = cv_deviance - cv_deviance_se,
            ymax = cv_deviance + cv_deviance_se
        ),
        width = 0.1
    ) +
    geom_line(
        data = df_plot,
        aes(
            x = num_predictors,
            y = cv_deviance
        ),
        color = "red",
        size = 1
    ) +
    geom_point(
        data = df_best,
        aes(
            x = num_predictors,
            y = cv_deviance,
            colour = "Best model"
        ),
        size = 2,
        alpha = 1
    ) +
    scale_colour_manual(
        values = c("Best model" = "blue")
    ) +
    scale_x_continuous(
        breaks = get_integer_breaks(df_plot$num_predictors)
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
            size = 14
        ),
        # legend.position = c(0.3, 0.1),
        legend.background = element_rect(
            fill = "transparent"
        )
    ) +
    xlab("Number of predictors") +
    ylab(y_var_name)

}


### Dimension reduction plot

get_integer_breaks = function(input_vector, interval = 1){
    minimum = floor(min(input_vector))
    maximum = ceiling(max(input_vector))
    breaks = seq(from = minimum,
                 to = maximum,
                 by = interval)
    return(breaks)
}

make_dim_reduc_plot = function(df_plot, df_best){
    
    ggplot() +
    geom_point(
        data = df_plot,
        aes(
            x = m,
            y = cv_mse
        ),
        color = "gray",
        size = 1,
        alpha = 0.7
    ) +
    geom_line(
        data = df_plot,
        aes(
            x = m,
            y = cv_mse
        ),
        color = "red",
        size = 1
    ) +
    geom_point(
        data = df_best,
        aes(
            x = best_m,
            y = cv_mse_best_m,
            colour = "Best Model"
        ),
        size = 2,
        alpha = 1
    ) +
    scale_y_log10() +
    scale_colour_manual(
        name = paste0("m = ", df_best$best_m[1]),
        values = c("Best Model" = "blue")
    ) +
    scale_x_continuous(
        breaks = get_integer_breaks(df_plot$m)
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
            size = 14
        ),
        legend.background = element_rect(
            fill = "transparent"
        )
    ) +
    xlab("Number of components") +
    ylab("CV MSE")
    
}


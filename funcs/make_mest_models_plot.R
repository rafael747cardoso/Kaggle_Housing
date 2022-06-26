
### Best models plot

make_mest_models_plot = function(df_models){
    ggplot() +
        geom_point(
            data = df_models,
            aes(
                x = models,
                y = cv_mse,
                color = models
            ),
            size = 3
        ) +
        geom_errorbar(
            data = df_models,
            aes(
                x = models,
                y = cv_mse,
                ymin = cv_mse - cv_mse_se,
                ymax = cv_mse + cv_mse_se,
                color = models
            ),
            width = 1
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
            legend.position = "none"
        ) +
        xlab("Model Selection type") +
        ylab("CV MSE")
}


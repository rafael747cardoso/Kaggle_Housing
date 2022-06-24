
### Box plot

make_boxplot = function(df_plot, x_var, y_var){
    
    require(stringr)
    
    x_var_name = x_var %>%
                     gsub(x = .,
                          pattern = "_",
                          replacement = " ") %>%
                     str_to_title()
    y_var_name = y_var %>%
                     gsub(x = .,
                          pattern = "_",
                          replacement = " ") %>%
                     str_to_title()
    
    df_plot = df_plot[, c(x_var, y_var)]
    sorted_levels = sort(unique(df_plot[, x_var]))
    df_plot[, x_var] = factor(x = df_plot[, x_var],
                              levels = sorted_levels)
    
    my_palette = colorRampPalette(c("#111539", "#97A1D9"))
    outlier_color = "#DA2E2E"
    median_color = "#23C16A"
    n_levels = length(unique(df_plot[, x_var]))
    plot_ly(
            data = df_plot,
            type = "box",
            y = ~eval(parse(text = y_var)),
            color = ~eval(parse(text = x_var)),
            colors = my_palette(n_levels),
            marker = list(
                color = outlier_color,
                opacity = 0.5,
                size = 5
            )
        ) %>%
        layout(
            xaxis = list(
                title = paste0("<b>", x_var_name, "</b>"),
                titlefont = list(
                    size = 20
                ),
                tickfont = list(
                    size = 18
                ),
                categoryorder = "array",
                autotick = FALSE
            ),
            yaxis = list(
                title = paste0("<b>", y_var_name, "</b>"),
                titlefont = list(
                    size = 20
                ),
                tickfont = list(
                    size = 18
                )
            ),
            hoverlabel = list(
                font = list(
                    size = 16
                )
            ),
            showlegend = FALSE
        )

}

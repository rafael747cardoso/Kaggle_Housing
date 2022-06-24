
### Density with mean and standard deviations

make_dens_plot = function(X, x_var_name = "x"){
    mean_X = mean(X)
    std_X = sd(X)
    dens = density(X)
    npoints = 2
    plot_ly() %>%
        add_trace(
            x = dens$x,
            y = dens$y,
            type = "scatter",
            mode = "lines",
            fill = "tozeroy",
            color = "#813DDA",
            colors = "#813DDA",
            opacity = 0.9,
            hovertemplate = paste0("<b>", x_var_name, ": %{x:,}<br>",
                                   "Density: %{y:,}</b><extra></extra>"),
            name = "Data"
        ) %>%
        add_trace(
            x = rep(mean_X + std_X, npoints),
            y = seq(from = min(dens$y),
                    to = 1.1*max(dens$y),
                    length.out = npoints),
            type = "scatter",
            mode = "lines",
            fill = "toself",
            fillcolor = "rgba(129, 198, 118, 0.5)",
            line = list(
                color = "#D33F1C"
            ),
            name = "Mean + Std"
        ) %>%
        add_trace(
            x = rep(mean_X - std_X, npoints),
            y = seq(from = min(dens$y),
                    to = 1.1*max(dens$y),
                    length.out = npoints),
            type = "scatter",
            mode = "lines",
            fill = "tonextx",
            fillcolor = "rgba(129, 198, 118, 0.5)",
            line = list(
                color = "#1C69D3"
            ),
            name = "Mean - Std"
        ) %>%
        add_trace(
            x = rep(mean_X, npoints),
            y = seq(from = min(dens$y),
                    to = 1.1*max(dens$y),
                    length.out = npoints),
            type = "scatter",
            mode = "lines",
            line = list(color = "#259A12"),
            name = "Mean"
        ) %>%
        layout(
            xaxis = list(title = paste0("<b>", x_var_name, "</b>")),
            yaxis = list(title = paste0("<b>Density</b>")),
            showlegend = TRUE
        )

}

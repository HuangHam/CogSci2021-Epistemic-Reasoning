# functions for data analysis
library(plotrix)

addSmallLegend <- function(myPlot, pointSize = 0.8, textSize = 11, spaceLegend = 2.5) {
  myPlot +
    guides(shape = guide_legend(override.aes = list(size = pointSize)),
           color = guide_legend(override.aes = list(size = pointSize))) +
    theme(legend.title = element_text(size = textSize), 
          legend.text  = element_text(size = textSize),
          axis.title=element_text(size=18),
          legend.key.size = unit(spaceLegend, "lines"))
}

# functions for calculating some statistics
se <- function(v,...){ # compute standard error
  plotrix::std.error(v,...)
}

# functions for data transformation
shapeMean <- function(df, ...){ # compute only mean by certain variable group
  df %>%
    group_by_at(c(...)) %>%
    summarise(across(where(is.numeric), ~ mean(.x, na.rm = T)))
}
shapeBoth <- function(df, ...){ # compute both mean and se by certain variable group
  df %>%
    group_by_at(c(...)) %>%
    summarise(across(where(is.numeric),list(m = mean, se = se), na.rm = T))
}

# ploting
lineplt <-function(df, x_var, y_var, y_se, condition, title, xlabel, ylabel) { # line plot
  x = substitute(x_var)
  y = substitute(y_var)
  condition = substitute(condition)
  ggplot(df, aes_(x=x, y=y, color = condition, group=condition, ymin=substitute(y_var-y_se), ymax=substitute(y_var+y_se))) + 
    geom_point()+
    geom_line() +
    ggtitle(title) +
    xlab(xlabel) + 
    ylab(ylabel) +
    geom_errorbar(width=.2,position=position_dodge(0))
}
addLine <- function(plot, y_var,y_se,y_name, dual_axis_scale = 1, legend_pos = c(0.8, 0.8), breaks = seq(0,10,2)){ # add line and axis to a existing line plt
  plot+
	geom_line(aes_(y = substitute(y_var/dual_axis_scale), colour = substitute(y_name)))+
  geom_errorbar(aes_(ymin=substitute((y_var-y_se)/dual_axis_scale), ymax=substitute((y_var+y_se)/dual_axis_scale), 
  	colour = substitute(y_name)),width=.2, position=position_dodge(0))+ 
  scale_y_continuous(sec.axis = sec_axis(~.*dual_axis_scale, name = y_name), breaks = breaks)+ 
  theme(legend.position = legend_pos,legend.background = element_rect(colour = "transparent", fill = "transparent"))
}


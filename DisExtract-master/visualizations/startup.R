# install.packages('devtools')
# devtools::install_github("mhtess/rwebppl")
# install.packages("ggplot2")
# install.packages("tidyr")
# install.packages("dplyr", dependencies=T)
# install.packages("jsonlite")
# install.packages("ggthemes")
# install.packages("lme4")
# install.packages("lmerTest")
# install.packages("memoise")
library(rwebppl)
library(ggplot2)
library(tidyr)
library(dplyr)
library(jsonlite)
library(ggthemes)
library(lme4)
library(lmerTest)
library(memoise)

print_colornames = function() {
  return(colors())
}

theme_black = theme_few(18) + theme(
  plot.background = element_rect(fill="black", colour="black"),
  plot.title = element_text(colour="lightgray"),
  axis.line.x = element_line(colour="lightgray"),
  axis.line.y = element_line(colour="lightgray"),
  axis.ticks = element_line(colour="lightgray"),
  axis.text = element_text(colour="lightgray"),
  legend.title = element_text(colour="lightgray"),
  legend.text = element_text(colour="lightgray"),
  legend.background = element_rect(fill="#333333"),
  strip.text = element_text(colour="lightgray"),
  panel.border = element_rect(colour="black"),
  panel.background = element_rect(fill="black"),
  strip.background = element_rect(fill="black"),
  # legend.background = element_rect(fill="black"),
  legend.key = element_rect(fill="#333333"),
  # legend.position = "none",
  axis.title = element_text(colour="lightgray"))

char = as.character
num = function(v) {return(as.numeric(as.character(v)))}

theme.new = theme_set(theme_few(12))

set_theme_black = function() {
  
  theme.new = theme_set(theme_black)
  
  scale_colour_black = function() {
    return(scale_colour_hue(c=55, l=35))
  }
  scale_fill_black = function() {
    return(scale_fill_hue(c=55, l=35))
  }
}

# for bootstrapping 95% confidence intervals
theta <- function(x,xdata) {mean(xdata[x])}
ci.low <- function(x) {
  quantile(bootstrap::bootstrap(1:length(x),1000,theta,x)$thetastar,.025)}
ci.high <- function(x) {
  quantile(bootstrap::bootstrap(1:length(x),1000,theta,x)$thetastar,.975)}

# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
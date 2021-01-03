library(readr)
library(partykit)
library(tidyr)
library(ggplot2)
set.seed(12345)

women <- read_delim("Women.csv", ";", col_types = cols())
colnames(women)[1] <- "Blood.Systolic"

reg_model <- function(y, x, start = NULL, weights = NULL, offset = NULL, ...) {
  
  glm(y ~ 0 + x, family = gaussian, start = start, ...)
  #loess(y ~ 0 + x, weights = weights, family = "gaussian", method = "loess", ...)
  
}

tree <- mob(Blood.Systolic ~ height + weight | height + weight, data = women,
                 fit = reg_model, control = mob_control(minsize = 5000))

plot(tree)

new_height <- seq(min(women$height), max(women$height), len = 100)
new_weight <- seq(min(women$weight), max(women$weight), len = 100)
data_for_pred <- crossing(height = new_height, weight = new_weight)

### taken from tree plot
data_for_pred$`Prediction (LM)` <- 
  222.8732 -0.477*data_for_pred$height + 0.5575*data_for_pred$weight 

loess_model <- loess(Blood.Systolic ~ height + weight,
                     family = "gaussian", method = "loess", data = women)

data_for_pred$`Prediction (LOESS)` <- predict(loess_model, data_for_pred)

ggplot(data_for_pred) + geom_raster(aes(height, weight, fill = `Prediction (LM)`)) +
  ggtitle("LM Predictions")

ggplot(data_for_pred) + 
  geom_raster(aes(height, weight, fill = `Prediction (LOESS)`)) +
  ggtitle("LOESS Predictions")
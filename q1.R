library(readr)
library(glmnet)
set.seed(12345)

parkinsons <- read_csv("parkinsons.csv", col_types = cols())
full_data <- scale(subset(parkinsons, select = -total_UPDRS))

y_full_data <- full_data[, "motor_UPDRS"]
x_full_data <- subset(full_data, select = -motor_UPDRS)

lambda <- seq(1, 0.1, -0.1)
alpha <- 0.5
pi_thr <- 0.7
B <- 100

col_weights <- runif(ncol(x_full_data), alpha, 1)
resampling_indices <- sapply(1:B, function(i) 
  sample(1:nrow(x_full_data), nrow(x_full_data), replace = TRUE))
models <- sapply(1:B, function(i) glmnet(x_full_data[resampling_indices[, i], ], 
                                         y_full_data[resampling_indices[, i]], 
                                         family = "gaussian", 
                                         lambda = seq(1, 0.1, -0.1), standardize = F,
                                         penalty.factor = 1/col_weights)$beta != 0)
num_times_selected <- sapply(1:length(lambda), function(i)
  sapply(1:ncol(x_full_data), function(j) sum(sapply(models, function(x) x[j, i]))))
row.names(num_times_selected) <- colnames(x_full_data)
colnames(num_times_selected) <- paste("lambda =", lambda)

pi_hat <- round(num_times_selected/100, 2)
View(pi_hat)
max_pi_hat <- sapply(1:nrow(pi_hat), function(x) max(pi_hat[x, ]))
S_hat_stable <- which(max_pi_hat > 0.7)
colnames(x_full_data)[S_hat_stable]
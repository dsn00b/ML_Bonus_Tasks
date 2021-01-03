set.seed(1234567890)

Var <- runif(50, 0, 10)
trva <- data.frame(Var, Sin=sin(Var))
tr <- trva[1:25,] # Training
va <- trva[26:50,] # Validation

# plot(trva)
# plot(tr)
# plot(va)

l_rate <- 1/nrow(tr)^2
n_ite = 25000

# modified to allow sigmoid/tanh solutions to be computed simultaneously

w_j_sigmoid <- runif(10, -1, 1)
b_j_sigmoid <- runif(10, -1, 1)
w_k_sigmoid <- runif(10, -1, 1)
b_k_sigmoid <- runif(1, -1, 1)

error_sigmoid <- rep(0, n_ite) 
error_va_sigmoid <- rep(0, n_ite)

w_j_tanh <- runif(10, -1, 1)
b_j_tanh <- runif(10, -1, 1)
w_k_tanh <- runif(10, -1, 1)
b_k_tanh <- runif(1, -1, 1)

error_tanh <- rep(0, n_ite)
error_va_tanh <- rep(0, n_ite)

for(i in 1:n_ite) {

  # set up data
  x_tr <- tr$Var
  y_tr <- tr$Sin
  
  # error computation: Your code here
  
  # forward propagation
  h1_sigmoid <- (x_tr %*% t(w_j_sigmoid)) + t(matrix(b_j_sigmoid, 10, nrow(tr)))
  h1_tanh <- (x_tr %*% t(w_j_tanh)) + t(matrix(b_j_tanh, 10, nrow(tr)))
  h1_sigmoid_va <- (va$Var %*% t(w_j_sigmoid)) + t(matrix(b_j_sigmoid, 10, nrow(va)))
  h1_tanh_va <- (va$Var %*% t(w_j_tanh)) + t(matrix(b_j_tanh, 10, nrow(va)))
  
  a1_sigmoid <- 1 / (1 + exp(-h1_sigmoid))
  a1_tanh <- 2 / (1 + exp(-2*h1_tanh)) - 1
  a1_sigmoid_va <- 1 / (1 + exp(-h1_sigmoid_va))
  a1_tanh_va <- 2 / (1 + exp(-2*h1_tanh_va)) - 1
  
  y_hat_sigmoid <- (a1_sigmoid %*% w_k_sigmoid) + b_k_sigmoid
  y_hat_tanh <- (a1_tanh %*% w_k_tanh) + b_k_tanh
  y_hat_sigmoid_va <- (a1_sigmoid_va %*% w_k_sigmoid) + b_k_sigmoid
  y_hat_tanh_va <- (a1_tanh_va %*% w_k_tanh) + b_k_tanh
  
  # SSE error computation
  error_sigmoid[i] <- sum((y_tr - y_hat_sigmoid)^2)
  error_va_sigmoid[i] <- sum((va$Sin - y_hat_sigmoid_va)^2)
  error_tanh[i] <- sum((y_tr - y_hat_tanh)^2)
  error_va_tanh[i] <- sum((va$Sin - y_hat_tanh_va)^2)
  
  # print
  cat("Iteration: ", i, 
      ", Sigmoid error: ", error_sigmoid[i]/2, ", Sigmoid error_va: ", error_va_sigmoid[i]/2, 
      ", Tanh error: ", error_tanh[i]/2, ", Tanh error_va: ", error_va_tanh[i]/2, "\n") 
  flush.console()
  
  for(n in 1:nrow(tr)) {
    
    # forward propagation: Your code here
    
    h1_sigmoid <- (x_tr[n] * w_j_sigmoid) + b_j_sigmoid
    h1_tanh <- (x_tr[n] * w_j_tanh) + b_j_tanh
    
    a1_sigmoid <- 1 / (1 + exp(-h1_sigmoid))
    a1_tanh <- 2 / (1 + exp(-2*h1_tanh)) - 1
    
    y_hat_sigmoid <- sum(a1_sigmoid * w_k_sigmoid) + b_k_sigmoid
    y_hat_tanh <- sum(a1_tanh * w_k_tanh) + b_k_tanh
    
    # backward propagation: Your code here
    
    ### ASSUMPTION: error(y_hat | y) = 1/2 * (y_hat - y)^2
    
    delta_error_sigmoid_b_k <- (y_hat_sigmoid - y_tr[n])
    delta_error_sigmoid_w_k <- delta_error_sigmoid_b_k*a1_sigmoid
    delta_error_sigmoid_b_j <- delta_error_sigmoid_b_k*w_k_sigmoid*a1_sigmoid*(1- a1_sigmoid)
    delta_error_sigmoid_w_j <- x_tr[n]*delta_error_sigmoid_b_j
    
    delta_error_tanh_b_k <- (y_hat_tanh - y_tr[n])
    delta_error_tanh_w_k <- delta_error_tanh_b_k*a1_tanh
    delta_error_tanh_b_j <- delta_error_tanh_b_k*w_k_tanh*(1 - a1_tanh^2)
    delta_error_tanh_w_j <- x_tr[n]*delta_error_tanh_b_j
    
    b_k_sigmoid <- b_k_sigmoid - l_rate*delta_error_sigmoid_b_k
    w_k_sigmoid <- w_k_sigmoid - l_rate*delta_error_sigmoid_w_k
    b_j_sigmoid <- b_j_sigmoid - l_rate*delta_error_sigmoid_b_j
    w_j_sigmoid <- w_j_sigmoid - l_rate*delta_error_sigmoid_w_j
    
    b_k_tanh <- b_k_tanh - l_rate*delta_error_tanh_b_k
    w_k_tanh <- w_k_tanh - l_rate*delta_error_tanh_w_k
    b_j_tanh <- b_j_tanh - l_rate*delta_error_tanh_b_j
    w_j_tanh <- w_j_tanh - l_rate*delta_error_tanh_w_j
    
  }
  
}

# print final weights and errors
w_j_sigmoid
b_j_sigmoid
w_k_sigmoid
b_k_sigmoid

w_j_tanh
b_j_tanh
w_k_tanh
b_k_tanh

plot(error_sigmoid/2, ylim = c(0, 5), col = "blue")
points(error_va_sigmoid/2, col = "red")
title("Sigmoid Activation: Training (Blue) and Validation (Red) Error")

plot(error_tanh/2, ylim = c(0, 5), col = "blue")
points(error_va_tanh/2, col = "red")
title("Tanh Activation: Training (Blue) and Validation (Red) Error")

# plot prediction on training data

h1_sigmoid <- (x_tr %*% t(w_j_sigmoid)) + t(matrix(b_j_sigmoid, 10, nrow(tr)))
h1_tanh <- (x_tr %*% t(w_j_tanh)) + t(matrix(b_j_tanh, 10, nrow(tr)))

a1_sigmoid <- 1 / (1 + exp(-h1_sigmoid))
a1_tanh <- 2 / (1 + exp(-2*h1_tanh)) - 1

y_hat_sigmoid <- (a1_sigmoid %*% w_k_sigmoid) + b_k_sigmoid
y_hat_tanh <- (a1_tanh %*% w_k_tanh) + b_k_tanh

plot(x_tr, y_tr, col = "red")
points(x_tr, y_hat_sigmoid, col = "blue")
title("Sigmoid Activation: Training Predictions (Blue) and Labels (Red)")

plot(x_tr, y_tr, col = "red")
points(x_tr, y_hat_tanh, col = "blue")
title("Tanh Activation: Training Predictions (Blue) and Labels (Red)")

# plot prediction on validation data

h1_sigmoid_va <- (va$Var %*% t(w_j_sigmoid)) + t(matrix(b_j_sigmoid, 10, nrow(va)))
h1_tanh_va <- (va$Var %*% t(w_j_tanh)) + t(matrix(b_j_tanh, 10, nrow(va)))

a1_sigmoid_va <- 1 / (1 + exp(-h1_sigmoid_va))
a1_tanh_va <- 2 / (1 + exp(-2*h1_tanh_va)) - 1

y_hat_sigmoid_va <- (a1_sigmoid_va %*% w_k_sigmoid) + b_k_sigmoid
y_hat_tanh_va <- (a1_tanh_va %*% w_k_tanh) + b_k_tanh

plot(va$Var, va$Sin, col = "red")
points(va$Var, y_hat_sigmoid_va, col = "blue")
title("Sigmoid Activation: Validation Predictions (Blue) and Labels (Red)")

plot(va$Var, va$Sin, col = "red")
points(va$Var, y_hat_tanh_va, col = "blue")
title("Tanh Activation: Validation Predictions (Blue) and Labels (Red)")

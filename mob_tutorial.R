data("PimaIndiansDiabetes", package = "mlbench")

library(partykit)
library(sandwich)

pid_formula <- diabetes ~ glucose | pregnant + pressure + triceps + 
  insulin + mass + pedigree + age

logit <- function(y, x, start = NULL, weights = NULL, offset = NULL, ...) {
  glm(y ~ 0 + x, family = binomial, start = start, ...)
}

pid_tree <- mob(pid_formula, data = PimaIndiansDiabetes, fit = logit)

pid_tree2 <- glmtree(diabetes ~ glucose | pregnant +
                       pressure + triceps + insulin + mass + pedigree + age,
                     data = PimaIndiansDiabetes, family = binomial)

pid_tree3 <- mob(pid_formula, data = PimaIndiansDiabetes,
                 fit = logit, control = mob_control(verbose = TRUE,
                                                    minsize = 50, 
                                                    maxdepth = 4, 
                                                    alpha = 0.9, 
                                                    prune = "BIC"))

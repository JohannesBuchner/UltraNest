library(reticulate) # install.packages("reticulate") if you don't have it

np = import("numpy")
un = import("ultranest")  # pip or conda install it if you don't have it

paramnames = c("a", "b", "c")

mytransform <- function(params) {
  params * 2 - 1
}

mylikelihood <- function(params) {
  centers = 0.1 * 1:length(paramnames)
  dim(centers) <- c(1, 3)
  L = -0.5 * apply((apply(params, 1, '-', centers) / 0.01)**2, MARGIN=2, sum)
  np$asarray(L)
}

sampler = un$ReactiveNestedSampler(paramnames, mylikelihood, transform=mytransform, vectorized=TRUE)
results = sampler$run()

# show samples:
pairs(results$samples, labels=paramnames)

# integral estimate:
print(paste("marginal likelihood estimate:", results$logz, " +- ", results$logzerr))

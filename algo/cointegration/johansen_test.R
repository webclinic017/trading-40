library("urca")

set.seed(123)

z <- rep(0, 10000)
for (i in 2:10000) z[i] <- z[i-1] + rnorm(1)

p <- q <- r <- rep(0, 10000)
p <- 0.3*z + rnorm(10000)
q <- 0.6*z + rnorm(10000)
r <- 0.2*z + rnorm(10000)

jotest = ca.jo(data.frame(p,q,r), type="trace", K=2, ecdet="none", spec="longrun")
summary(jotest)

s = 1.000*p + 1.791324*q - 1.717271*r
plot(s, type="l")


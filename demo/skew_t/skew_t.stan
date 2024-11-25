functions {
    #include skew_generalized_t.stanfunctions
}

data {
    int<lower=0> N;

    real mu;
    real<lower=0> sigma;
    real<lower=-1, upper=1> lambda;
    real<lower=0> p;
    real<lower=0> q;
}

parameters {
    vector[N] x;
}

model {
    target += skew_generalized_t_lpdf(x | mu, sigma, lambda, p, q);
}

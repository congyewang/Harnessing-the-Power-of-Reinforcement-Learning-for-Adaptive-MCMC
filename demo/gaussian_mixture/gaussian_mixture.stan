data {
  real<lower=0> weight;

  real mu1;
  real mu2;

  real<lower=0> sigma1;
  real<lower=0> sigma2;
}

parameters {
  real x;
}

model {
  target += log_mix(weight,
                    normal_lpdf(x | mu1, sigma1),
                    normal_lpdf(x | mu2, sigma2)
                  );
}

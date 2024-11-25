data {
  real<lower=0> mu;
  real<lower=0> sigma2;
}

parameters {
  vector[2] x;
}

model {
  target += -x[1]^2 / (2 * sigma2);
  target += -((x[2] - (x[1]^2 + mu))^2) / (2 * sigma2);
}


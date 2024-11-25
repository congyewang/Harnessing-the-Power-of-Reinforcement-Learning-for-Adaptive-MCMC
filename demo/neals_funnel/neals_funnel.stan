parameters {
  real x;
  real y;
}

model {
  y ~ normal(0, 3);
  x ~ normal(0, exp(y / 2));
}


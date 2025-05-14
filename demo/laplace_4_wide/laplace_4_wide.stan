
data {
  int<lower=1> N;
  real r;
}

parameters {
  vector[N] x;
}

model {
  target += -norm2(x)^r / 10.0;
}


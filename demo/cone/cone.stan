functions {
  real norm(vector x) {
    real result = 0;
    for (i in 1:size(x)) {
      result += x[i]^2;
    }
    return sqrt(result);
  }
}

data {
  int<lower=0> N;
  real<lower=0> beta;
}

parameters {
  vector[N] x;
}

model {
  target += -norm(x)^beta;
}

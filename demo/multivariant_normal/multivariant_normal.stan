data {
  int<lower=1> N;

  vector[N] mean;

  matrix[N, N] cov;
}

parameters {
    vector[2] x;
}

model {
    x ~ multi_normal(mean, cov);
}

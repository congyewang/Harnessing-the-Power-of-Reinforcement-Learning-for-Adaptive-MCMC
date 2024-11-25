data {
  real a;
  real b;
}

parameters {
  vector[2] x;
}

model {
  target += -log(1.0 + (a - x[1])^2 + b * (x[2] - x[1]^2)^2);
}


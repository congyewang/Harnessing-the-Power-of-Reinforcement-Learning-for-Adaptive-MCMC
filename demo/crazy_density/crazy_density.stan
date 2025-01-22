functions {
  /**
   * Multiquadratic function applied element-wise in 2D.
   * (c^2 + x[i]^2)^b
   */
  vector multiquadratic(vector x, real b, real c) {
    vector[2] out;
    for (i in 1:2) {
      out[i] = pow(c^2 + square(x[i]), b);
    }
    return out;
  }

  /**
   * log_den_imbalanced:
   *   - sum( multiquadratic(scale_matrix * x, power, bias) )
   * scale_matrix = diag_matrix([1/scale, 1]).
   */
  real log_den_imbalanced(vector x, real power, real scale, real bias) {
    matrix[2, 2] scale_matrix;
    // Construct the diagonal matrix
    scale_matrix[1,1] = 1.0 / scale;
    scale_matrix[1,2] = 0.0;
    scale_matrix[2,1] = 0.0;
    scale_matrix[2,2] = 1.0;

    vector[2] scaled_x = scale_matrix * x;
    return -sum(multiquadratic(scaled_x, power, bias));
  }

  /**
   * log_den_imbalanced_rotate:
   *   Rotate x by 'angle', then call log_den_imbalanced.
   */
  real log_den_imbalanced_rotate(vector x, real angle, real power, real scale) {
    matrix[2, 2] rotate_matrix;
    rotate_matrix[1,1] = cos(angle);
    rotate_matrix[1,2] = sin(angle);
    rotate_matrix[2,1] = -sin(angle);
    rotate_matrix[2,2] = cos(angle);

    vector[2] x_rot = rotate_matrix * x;
    // bias = 0.01 by default
    return log_den_imbalanced(x_rot, power, scale, 0.01);
  }

  /**
   * log_den_n_mixture:
   *   sum of exp(log_den_imbalanced_rotate(...)) over i in [0, n-1],
   *   then take log of that sum.
   */
  real log_den_n_mixture(vector x, int n) {
    real sum_exp_terms = 0.0;

    for (i in 1:n) {
      real i0    = i - 1;              // so i0 goes 0..(n-1)
      real power = pow(0.4, i0 + 1);   // 0.4^(i0 + 1)
      real angle = i0 * pi() / 4.0;    // i0 * pi/4
      real scale = pow(1.5, n - i0);   // 1.5^(n - i0)
      sum_exp_terms += exp(log_den_imbalanced_rotate(x, angle, power, scale));
    }

    return log(sum_exp_terms);
  }
}

data {
  int<lower=1> n;
}

parameters {
  vector[2] x;
}

model {
  target += log_den_n_mixture(x, n);
}

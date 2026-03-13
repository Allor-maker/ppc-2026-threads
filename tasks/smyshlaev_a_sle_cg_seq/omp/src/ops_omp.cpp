#include "smyshlaev_a_sle_cg_seq/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "smyshlaev_a_sle_cg_seq/common/include/common.hpp"

namespace smyshlaev_a_sle_cg_seq {

SmyshlaevASleCgTaskOMP::SmyshlaevASleCgTaskOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SmyshlaevASleCgTaskOMP::ValidationImpl() {
  const auto &a = GetInput().A;
  const auto &b = GetInput().b;
  if (a.empty() || b.empty()) {
    return false;
  }
  if (a.size() != b.size()) {
    return false;
  }
  if (a.size() != a[0].size()) {
    return false;
  }
  return true;
}

bool SmyshlaevASleCgTaskOMP::PreProcessingImpl() {
  const auto &a = GetInput().A;
  size_t n = a.size();
  flat_A_.resize(n * n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      flat_A_[(i * n) + j] = a[i][j];
    }
  }
  return true;
}

bool SmyshlaevASleCgTaskOMP::RunImpl() {
  const auto &b = GetInput().b;
  int n = static_cast<int>(b.size());

  if (n == 0) {
    return true;
  }

  std::vector<double> r = b;
  std::vector<double> p = r;
  std::vector<double> ap(n, 0.0);
  std::vector<double> result(n, 0.0);

  double rs_old = 0.0;
  for (int i = 0; i < n; ++i) {
    rs_old += r[i] * r[i];
  }

  const int max_iterations = n * 2;
  const double epsilon = 1e-9;

  if (std::sqrt(rs_old) < epsilon) {
    GetOutput() = result;
    return true;
  }

  // Делаем локальную ссылку на поле класса, чтобы MSVC разрешил передать её в shared()
  const std::vector<double> &local_A = flat_A_;

  double p_ap = 0.0;
  double rs_new = 0.0;
  double alpha = 0.0;
  double beta = 0.0;
  bool converged = false;

  // Один parallel region. Никаких утечек памяти (Valgrind) и максимальная скорость.
#pragma omp parallel default(none) \
    shared(n, r, p, ap, result, local_A, rs_old, p_ap, rs_new, alpha, beta, converged, max_iterations, epsilon)
  {
    for (int iter = 0; iter < max_iterations; ++iter) {
#pragma omp single
      p_ap = 0.0;

      // 1. Вычисляем A * p
#pragma omp for schedule(static)
      for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
          sum += local_A[(i * n) + j] * p[j];
        }
        ap[i] = sum;
      }

      // 2. Скалярное произведение (с легальным reduction)
#pragma omp for schedule(static) reduction(+ : p_ap)
      for (int i = 0; i < n; ++i) {
        p_ap += p[i] * ap[i];
      }

#pragma omp single
      {
        if (std::abs(p_ap) < 1e-15) {
          converged = true;
        } else {
          alpha = rs_old / p_ap;
        }
        rs_new = 0.0;
      }

      if (converged) {
        break;
      }

      // 3. Обновление векторов и расчет новой невязки (с reduction)
#pragma omp for schedule(static) reduction(+ : rs_new)
      for (int i = 0; i < n; ++i) {
        result[i] += alpha * p[i];
        r[i] -= alpha * ap[i];
        rs_new += r[i] * r[i];
      }

#pragma omp single
      {
        if (std::sqrt(rs_new) < epsilon) {
          converged = true;
        } else {
          beta = rs_new / rs_old;
          rs_old = rs_new;
        }
      }

      if (converged) {
        break;
      }

      // 4. Обновление направления
#pragma omp for schedule(static)
      for (int i = 0; i < n; ++i) {
        p[i] = r[i] + (beta * p[i]);
      }
    }
  }

  GetOutput() = result;
  return true;
}

bool SmyshlaevASleCgTaskOMP::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_sle_cg_seq

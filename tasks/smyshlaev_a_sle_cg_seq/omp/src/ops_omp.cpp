#include "smyshlaev_a_sle_cg_seq/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "smyshlaev_a_sle_cg_seq/common/include/common.hpp"

namespace smyshlaev_a_sle_cg_seq {

namespace {

// Функции вызываются ИЗНУТРИ многопоточной зоны, поэтому внутри них только `omp for`
void ComputeAp_OMP(const std::vector<double> &matrix, const std::vector<double> &p, std::vector<double> &ap, size_t n) {
#pragma omp for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      sum += matrix[(i * n) + j] * p[j];
    }
    ap[i] = sum;
  }
}

// Используем классический reduction для переменной result_val (которая передается по ссылке)
void ComputeDotProduct_OMP(const std::vector<double> &v1, const std::vector<double> &v2, double &result_val, size_t n) {
#pragma omp for schedule(static) reduction(+ : result_val)
  for (size_t i = 0; i < n; ++i) {
    result_val += v1[i] * v2[i];
  }
}

void UpdateResultAndResidual_OMP(std::vector<double> &result_vec, std::vector<double> &r, const std::vector<double> &p,
                                 const std::vector<double> &ap, double alpha, double &rs_new, size_t n) {
#pragma omp for schedule(static) reduction(+ : rs_new)
  for (size_t i = 0; i < n; ++i) {
    result_vec[i] += alpha * p[i];
    r[i] -= alpha * ap[i];
    rs_new += r[i] * r[i];
  }
}

void UpdateP_OMP(std::vector<double> &p, const std::vector<double> &r, double beta, size_t n) {
#pragma omp for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    p[i] = r[i] + (beta * p[i]);
  }
}

}  // namespace

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
  size_t n = b.size();

  if (n == 0) {
    return true;
  }

  std::vector<double> r = b;
  std::vector<double> p = r;
  std::vector<double> ap(n, 0.0);
  std::vector<double> result(n, 0.0);

  // Вычисляем начальное rs_old последовательно, до старта потоков
  double rs_old = 0.0;
  for (size_t i = 0; i < n; ++i) {
    rs_old += r[i] * r[i];
  }

  const int max_iterations = static_cast<int>(n) * 2;
  const double epsilon = 1e-9;

  if (std::sqrt(rs_old) < epsilon) {
    GetOutput() = result;
    return true;
  }

  // Общие (shared) переменные для синхронизации и хранения промежуточных вычислений
  double p_ap = 0.0;
  double rs_new = 0.0;
  double alpha = 0.0;
  double beta = 0.0;
  bool converged = false;

  // ОДИН вызов создания пула потоков на весь метод (снимает проблемы с утечками в libomp)
#pragma omp parallel default(none) \
    shared(n, r, p, ap, result, flat_A_, rs_old, p_ap, rs_new, alpha, beta, converged, max_iterations, epsilon)
  {
    for (int iter = 0; iter < max_iterations; ++iter) {
      // Только один поток обнуляет переменную перед новой итерацией (остальные ждут на барьере)
#pragma omp single
      p_ap = 0.0;

      ComputeAp_OMP(flat_A_, p, ap, n);
      // Внутри используется reduction, который соберет суммы со всех потоков в p_ap
      ComputeDotProduct_OMP(p, ap, p_ap, n);

#pragma omp single
      {
        if (std::abs(p_ap) < 1e-15) {
          converged = true;
        } else {
          alpha = rs_old / p_ap;
        }
        rs_new = 0.0;  // Обнуляем перед следующим reduction
      }

      // Все потоки проверяют флаг, и если нужно — выходят
      if (converged) {
        break;
      }

      UpdateResultAndResidual_OMP(result, r, p, ap, alpha, rs_new, n);

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

      UpdateP_OMP(p, r, beta, n);
    }
  }

  GetOutput() = result;
  return true;
}

bool SmyshlaevASleCgTaskOMP::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_sle_cg_seq

#include "smyshlaev_a_sle_cg_seq/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "smyshlaev_a_sle_cg_seq/common/include/common.hpp"

namespace smyshlaev_a_sle_cg_seq {

namespace {

// Последовательные вспомогательные функции (O(N)) - просто и надежно
double ComputeDotProduct(const std::vector<double> &v1, const std::vector<double> &v2) {
  double res = 0.0;
  const size_t n = v1.size();
  for (size_t i = 0; i < n; ++i) {
    res += v1[i] * v2[i];
  }
  return res;
}

double UpdateRes(std::vector<double> &x, std::vector<double> &r, const std::vector<double> &p,
                 const std::vector<double> &ap, double alpha) {
  double rs_new = 0.0;
  const size_t n = x.size();
  for (size_t i = 0; i < n; ++i) {
    x[i] += alpha * p[i];
    r[i] -= alpha * ap[i];
    rs_new += r[i] * r[i];
  }
  return rs_new;
}

// Параллельное умножение матрицы на вектор (O(N^2))
// Используем сырые указатели, чтобы Valgrind и MSVC не ругались
void MultiplyParallel(const double *mat, const double *p, double *ap, int n) {
#pragma omp parallel for default(none) shared(mat, p, ap, n) schedule(static)
  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      sum += mat[i * n + j] * p[j];
    }
    ap[i] = sum;
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
  return !a.empty() && a.size() == b.size() && a.size() == a[0].size();
}

bool SmyshlaevASleCgTaskOMP::PreProcessingImpl() {
  const auto &a = GetInput().A;
  size_t n = a.size();
  flat_A_.resize(n * n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      flat_A_[i * n + j] = a[i][j];
    }
  }
  return true;
}

bool SmyshlaevASleCgTaskOMP::RunImpl() {
  const auto &b = GetInput().b;
  const int n = static_cast<int>(b.size());
  if (n == 0) {
    return true;
  }

  std::vector<double> r = b;
  std::vector<double> p = r;
  std::vector<double> ap(n, 0.0);
  std::vector<double> result(n, 0.0);

  double rs_old = ComputeDotProduct(r, r);
  const double eps = 1e-9;

  if (std::sqrt(rs_old) < eps) {
    GetOutput() = result;
    return true;
  }

  // Получаем указатели заранее, чтобы не дергать .data() в цикле
  const double *a_ptr = flat_A_.data();
  double *p_ptr = p.data();
  double *ap_ptr = ap.data();

  for (int iter = 0; iter < n * 2; ++iter) {
    // Параллельный запуск самой тяжелой части
    MultiplyParallel(a_ptr, p_ptr, ap_ptr, n);

    double p_ap = ComputeDotProduct(p, ap);
    if (std::abs(p_ap) < 1e-15) {
      break;
    }

    double alpha = rs_old / p_ap;
    double rs_new = UpdateRes(result, r, p, ap, alpha);

    if (std::sqrt(rs_new) < eps) {
      break;
    }

    double beta = rs_new / rs_old;
    // Обновляем p последовательно
    for (int i = 0; i < n; ++i) {
      p[i] = r[i] + beta * p[i];
    }

    rs_old = rs_new;
  }

  GetOutput() = result;
  return true;
}

bool SmyshlaevASleCgTaskOMP::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_sle_cg_seq

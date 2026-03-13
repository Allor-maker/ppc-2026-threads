#include "smyshlaev_a_sle_cg_seq/omp/include/ops_omp.hpp"

#include <atomic>
#include <numeric>
#include <vector>

#include "smyshlaev_a_sle_cg_seq/common/include/common.hpp"
#include "util/include/util.hpp"

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

  return true;
}

bool SmyshlaevASleCgTaskOMP::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_sle_cg_seq

#pragma once

#include "smyshlaev_a_conj_grad_seq/common/include/common.hpp"
#include "task/include/task.hpp"

namespace smyshlaev_a_conj_grad_seq {

class SmyshlaevAConjGradTaskOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }
  explicit SmyshlaevAConjGradTaskOMP(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace smyshlaev_a_conj_grad_seq

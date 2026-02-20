#pragma once

#include "smyshlaev_a_conj_grad_seq/common/include/common.hpp"
#include "task/include/task.hpp"

namespace smyshlaev_a_conj_grad_seq {

class SmyshlaevAConjGradTaskTBB : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kTBB;
  }
  explicit SmyshlaevAConjGradTaskTBB(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace smyshlaev_a_conj_grad_seq

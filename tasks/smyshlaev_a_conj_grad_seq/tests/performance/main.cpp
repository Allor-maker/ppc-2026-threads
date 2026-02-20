#include <gtest/gtest.h>

#include "smyshlaev_a_conj_grad_seq/all/include/ops_all.hpp"
#include "smyshlaev_a_conj_grad_seq/common/include/common.hpp"
#include "smyshlaev_a_conj_grad_seq/omp/include/ops_omp.hpp"
#include "smyshlaev_a_conj_grad_seq/seq/include/ops_seq.hpp"
#include "smyshlaev_a_conj_grad_seq/stl/include/ops_stl.hpp"
#include "smyshlaev_a_conj_grad_seq/tbb/include/ops_tbb.hpp"
#include "util/include/perf_test_util.hpp"

namespace smyshlaev_a_conj_grad_seq {

class SmyshlaevARunPerfTestThreads : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 200;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SmyshlaevARunPerfTestThreads, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SmyshlaevAConjGradTaskALL, SmyshlaevAConjGradTaskOMP, SmyshlaevAConjGradTaskSEQ,
                                SmyshlaevAConjGradTaskSTL, SmyshlaevAConjGradTaskTBB>(PPC_SETTINGS_smyshlaev_a_conj_grad_seq);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SmyshlaevARunPerfTestThreads::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SmyshlaevARunPerfTestThreads, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace smyshlaev_a_conj_grad_seq

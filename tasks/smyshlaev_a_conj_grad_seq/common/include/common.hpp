#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace smyshlaev_a_conj_grad_seq {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace smyshlaev_a_conj_grad_seq

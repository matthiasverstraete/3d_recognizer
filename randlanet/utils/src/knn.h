#pragma once
#include <torch/extension.h>
std::pair<at::Tensor, at::Tensor> knn(at::Tensor support, at::Tensor query, int k);

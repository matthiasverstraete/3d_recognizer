#include "knn.h"

using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn", &knn, "", "support"_a, "querry"_a, "k"_a);
}

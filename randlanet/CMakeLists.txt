# find python executable and include dirs
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

# find torch include and library dirs
exec_program(${PYTHON_EXECUTABLE}
    ARGS "-c \"import sys, os; sys.stdout = open(os.devnull, 'w'); from torch.utils import cpp_extension; sys.stdout = sys.__stdout__; print(';'.join(cpp_extension.include_paths()))\""
    OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
    RETURN_VALUE TORCH_NOT_FOUND)
if(TORCH_NOT_FOUND)
    message(FATAL_ERROR "Torch headers not found!")
endif()
exec_program(${PYTHON_EXECUTABLE}
    ARGS "-c \"import sys, os; sys.stdout = open(os.devnull, 'w'); from torch.utils import cpp_extension; sys.stdout = sys.__stdout__; print(';'.join(cpp_extension.library_paths()))\""
    OUTPUT_VARIABLE TORCH_LIBRARY_DIRS
    RETURN_VALUE TORCH_NOT_FOUND)
if(TORCH_NOT_FOUND)
    message(FATAL_ERROR "Torch libraries not found!")
endif()
link_directories(${TORCH_LIBRARY_DIRS})

add_library(knn_tpk SHARED utils/src/bindings.cpp utils/src/knn.cpp)
target_compile_options(knn_tpk PRIVATE -O3 -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE="_gcc" -DTORCH_EXTENSION_NAME=knn_tpk -DPYBIND11_STDLIB="_libstdcpp" -DPYBIND11_BUILD_ABI="_cxxabi1011" -D_GLIBCXX_USE_CXX11_ABI=0)
set_target_properties(knn_tpk PROPERTIES PREFIX "" LINKER_LANGUAGE C LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/utils/)
target_include_directories(knn_tpk PRIVATE utils/src ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(knn_tpk -ltorch -ltorch_cpu -ltorch_python)



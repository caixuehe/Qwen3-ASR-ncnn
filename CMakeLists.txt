cmake_minimum_required(VERSION 3.16)
project(qwen3_asr_ncnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/utf-8 /O2)
else()
  add_compile_options(-O3)
endif()

find_package(OpenMP)

# Pure-C++ reference executable (audio -> text) + per-stage test harnesses.
# These verify every stage numerically against the PyTorch golden and need no ncnn.
set(PURE qwen3_asr mel_features asr_frontend asr_encoder asr_thinker asr_thinker_kv asr_p4)
foreach(t ${PURE})
  add_executable(${t} src/${t}.cpp)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${t} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endforeach()

# ncnn-integrated build: the conv front-end runs through ncnn::Net (pnnx-converted).
# Enable by pointing NCNN_INSTALL at an ncnn install prefix, e.g.
#   cmake -S . -B build -DNCNN_INSTALL=third_party/ncnn/build/install
if(NCNN_INSTALL)
  find_package(ncnn REQUIRED PATHS "${NCNN_INSTALL}/lib/cmake/ncnn" NO_DEFAULT_PATH)
  foreach(t qwen3_asr_ncnn asr_frontend_ncnn)
    add_executable(${t} src/${t}.cpp)
    target_link_libraries(${t} PRIVATE ncnn)
    if(OpenMP_CXX_FOUND)
      target_link_libraries(${t} PRIVATE OpenMP::OpenMP_CXX)
    endif()
  endforeach()
  message(STATUS "ncnn-integrated targets enabled (qwen3_asr_ncnn, asr_frontend_ncnn)")
endif()

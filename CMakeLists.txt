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

# End-to-end pure-C++ executable: audio waveform -> text.
add_executable(qwen3_asr src/qwen3_asr.cpp)
if(OpenMP_CXX_FOUND)
  target_link_libraries(qwen3_asr PRIVATE OpenMP::OpenMP_CXX)
endif()

# Per-component test harnesses (each verifies one stage against PyTorch golden).
foreach(t mel_features asr_frontend asr_encoder asr_thinker asr_p4)
  add_executable(${t} src/${t}.cpp)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${t} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endforeach()

cmake_minimum_required(VERSION 3.5)
project(decal VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

option(USE_CUDA "Enable CUDA support (paper path). Set OFF for a best-effort CPU build." ON)
if (NOT APPLE AND USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
else ()
    set(USE_CUDA OFF)
endif ()

set(ONNXRUNTIME_VERSION 1.22.0)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}")
if (NOT EXISTS ${ONNXRUNTIME_ROOT})
    message(FATAL_ERROR
        "ONNX Runtime GPU ${ONNXRUNTIME_VERSION} not found at ${ONNXRUNTIME_ROOT}. "
        "From the repository root run: bash scripts/download_ort.sh")
endif ()
include_directories(${ONNXRUNTIME_ROOT}/include)

find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})

add_executable(decal_offline
    main.cpp
    inference.cpp inference.h
    sei_parser.cpp sei_parser.h
    offline_processor.cpp offline_processor.h
)
add_executable(decal_baseline
    main_current_baseline.cpp
    inference.cpp inference.h
    sei_parser.cpp sei_parser.h
    offline_processor_current_baseline.cpp offline_processor_current_baseline.h
)
add_executable(decal_duo
    main_duo.cpp
    inference.cpp inference.h
    sei_parser.cpp sei_parser.h
    offline_processor.cpp offline_processor.h
    offline_processor_duo.cpp offline_processor_duo.h
)
add_executable(decal_server
    server/online_server.cpp
    inference.cpp inference.h
    sei_parser.cpp sei_parser.h
    offline_processor.cpp offline_processor.h
)
add_executable(decal_client
    client/online_client.cpp
    sei_parser.cpp sei_parser.h
    template_ipc.cpp template_ipc.h
)
add_executable(decal_template_server
    server/template_server.cpp
    template_ipc.cpp template_ipc.h
)

foreach(tgt IN ITEMS decal_offline decal_baseline decal_duo decal_server)
    target_link_libraries(${tgt}
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
        OpenSSL::SSL OpenSSL::Crypto)
    if (USE_CUDA)
        target_link_libraries(${tgt}
            ${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_cuda.so
            ${CUDA_LIBRARIES})
    endif ()
endforeach()
target_link_libraries(decal_client OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(decal_template_server OpenSSL::SSL OpenSSL::Crypto)

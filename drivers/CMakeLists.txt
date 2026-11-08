add_executable(repair-stage "${CMAKE_CURRENT_LIST_DIR}/stage.cpp")
target_link_libraries(repair-stage PRIVATE trellis_core)
add_executable(repair-probe "${CMAKE_CURRENT_LIST_DIR}/probe.cpp")
target_link_libraries(repair-probe PRIVATE trellis_core ggml-vulkan)

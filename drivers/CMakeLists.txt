add_executable(repair-stage "${CMAKE_CURRENT_LIST_DIR}/stage.cpp")
target_link_libraries(repair-stage PRIVATE trellis_core)
add_executable(repair-probe "${CMAKE_CURRENT_LIST_DIR}/probe.cpp")
target_link_libraries(repair-probe PRIVATE trellis_core ggml-vulkan)
add_executable(repair-conditioning "${CMAKE_CURRENT_LIST_DIR}/conditioning.cpp")
target_link_libraries(repair-conditioning PRIVATE trellis_core ggml-vulkan)
# Upstream test_naf.cpp uses M_PI. MSVC exposes it only when requested before cmath.
# Test-only build integration; neither runtime variant's algorithms are changed.
if(MSVC)
  target_compile_definitions(trellis-test-naf PRIVATE _USE_MATH_DEFINES)
endif()

add_executable(repair-decoder "${CMAKE_CURRENT_LIST_DIR}/decoder.cpp")
target_link_libraries(repair-decoder PRIVATE trellis_core)

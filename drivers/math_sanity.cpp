// MSVC build-only smoke check, before SDK/shader acquisition.
#ifndef _USE_MATH_DEFINES
#error _USE_MATH_DEFINES must be set by the compiler before math headers
#endif
#include <cmath>
#ifndef M_PI
#error MSVC math constants are unavailable with the selected compiler options
#endif
static_assert(M_PI > 3.14159 && M_PI < 3.14160, "Math constant unavailable");
int main() { return std::isfinite(M_PI) ? 0 : 1; }

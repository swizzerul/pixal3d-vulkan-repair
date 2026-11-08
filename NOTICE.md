# Notices

The two patches derive from the pinned Pixal3D and GGML sources and retain their respective upstream MIT terms. Build scripts and new diagnostic drivers in this repository are provided under the MIT license in LICENSE.

Runtime bundles include the pinned native project's LICENSE and vendored third-party license material, GGML LICENSE, libwebp COPYING/PATENTS/AUTHORS and SPIRV-Headers license material. The native source contains stb, xatlas, meshoptimizer, fqms, httplib and other vendored components; their embedded notices are retained in the bundle's notices directory where applicable.

Source attribution:

- https://github.com/raven38/pixal3d.cpp
- https://github.com/pwilkin/ggml
- https://github.com/webmproject/libwebp
- https://github.com/KhronosGroup/SPIRV-Headers

Vulkan SDK build tools are supplied by LunarG and used only in the ephemeral build environment under their published terms. No SDK or Microsoft compiler payload is redistributed in the runtime bundle. The Windows Vulkan loader comes from the user's existing GPU driver.

Experimental build checksums identify a particular published build; they are not a guarantee against every supply-chain risk or a certification of model accuracy.

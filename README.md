# Experimental Pixal3D Vulkan repair build

This tools-only repository builds a same-toolchain unpatched control and a candidate containing two narrowly scoped repairs: actual-shape NAF convolution routing and Vulkan conversion-cache invalidation after shared scratch-buffer writers. It is not a production release or a claim of numerical equivalence.

Public sources are pinned in `pins.json`. The baseline and candidate use identical diagnostic logging and additive test-driver integration. Only the candidate receives the two patches. Complete bundles must remain separate; do not swap individual DLLs into another installation.

The workflow is manual only, uses the standard `windows-2022` runner, and runs no model inference. It downloads no images, weights or private fixtures. It publishes prerelease bundles, checksums and a build manifest using the repository-scoped job token. No Actions artifacts, persistent cache, Packages or LFS are used.

The bounded campaign permits one initial execution and at most one diagnosed, corrected build execution: 120 minutes per execution, 180 aggregate. The dispatch input must fit the remaining aggregate allowance. A cancelled or failed execution still consumes its elapsed time. No unchanged rerun is permitted.

`repair-probe --cpu` performs a tiny CPU arithmetic check and prints the compiled tensor ABI and operation enumeration. `repair-probe --route` builds synthetic NAF metadata without allocating or computing the represented graph. `repair-stage` exposes only the pinned image preprocessing, DINO and NAF stages and accepts local paths. These tools do not establish Vulkan correctness by compiling successfully.

Local verification must separately demonstrate both cache-writer paths, represented-input convolution accuracy, actual-shape support, synchronous native guard coverage and genuine affected-stage execution before model generation. Direct convolution can use different accumulation precision. Finite output alone does not prove numerical parity.

The Vulkan SDK is acquired and checked in the ephemeral runner only. Runtime bundles do not include the compiler, SDK, driver or Microsoft redistributable installer. A compatible system Vulkan loader/driver and Microsoft C++ runtime are runtime prerequisites.

See `NOTICE.md` and the bundled upstream notices for licensing and attribution.

Campaign status: both authorized executions failed; no runtime bundles or releases exist. The first run stopped at patch hashes because Windows checkout converted line endings. The corrected checkout passed those checks; the second run reached baseline compilation and stopped because the upstream NAF CPU test did not enable MSVC's `M_PI` constants. A test-only `_USE_MATH_DEFINES` definition is now prepared, but has **not been built**. No further dispatch is authorized by this campaign.

`drivers/cache_regression.py` is a prepared, **unexecuted** generic local test driver. It requires an external guard and exact bundle manifest, uses one graph per scratch writer, and records represented-input arithmetic separately from dispatch-path coverage. It does not grant permission to run a campaign or establish that either repair works.

Continuation authorized 2026-11-08: the historical two-run closure above is preserved. Up to six additional diagnosed manual executions are now authorized, each at most 60 minutes and within the original aggregate 180 minutes (411 seconds previously charged). Stop after successful complete baseline/candidate publication. Both variants retain the identical target-scoped MSVC test definition; an early math-header compiler check and recorded verbose compiler commands verify it. Local numerical, path-coverage and model gates remain separate and unverified.

Continuation run 42505921525 compiled and linked every baseline target with the recorded math definition. The first CLI startup check stopped at Windows loader status `0xC0000135`; candidate compilation was not reached. The hosted prerequisites now include the checksum-pinned, same-version LunarG runtime component ZIP solely to expose its signed x64 Vulkan loader on the job process PATH. It is not a GPU driver, is not installed on the owner PC, and is excluded from both release bundles. Early DLL load checks precede costly compilation. Existing source/SDK versions, numerical patches and mandatory CPU checks are unchanged.

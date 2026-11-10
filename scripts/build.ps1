$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:GITHUB_ACTIONS -ne 'true' -or $env:GITHUB_REPOSITORY -ne 'swizzerul/pixal3d-vulkan-repair') { throw 'Ephemeral authorized runner only' }
if ($env:GH_TOKEN -or $env:GITHUB_TOKEN) { throw 'Publication credential must not enter dependency/build steps' }
$tools = $PWD.Path
$pins = Get-Content -LiteralPath "$tools/pins.json" -Raw | ConvertFrom-Json
$work = Join-Path $env:RUNNER_TEMP 'repair-build'
if (Test-Path -LiteralPath $work) { throw 'Fresh build directory required' }
New-Item -ItemType Directory -Path $work | Out-Null
$records = Join-Path $work 'records'
New-Item -ItemType Directory -Path $records | Out-Null
function Invoke-Checked([string]$Exe,[string[]]$Arguments) {
    Write-Host ('> ' + $Exe + ' ' + ($Arguments -join ' '))
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed: $LASTEXITCODE" }
}
function Fetch-Pinned($Spec,[string]$Directory) {
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
    Invoke-Checked git @('init',$Directory)
    Invoke-Checked git @('-C',$Directory,'config','core.autocrlf','false')
    Invoke-Checked git @('-C',$Directory,'remote','add','origin',('https://github.com/'+$Spec.repository+'.git'))
    Invoke-Checked git @('-C',$Directory,'-c','credential.helper=','fetch','--depth=1','origin',$Spec.commit)
    Invoke-Checked git @('-C',$Directory,'checkout','--detach','FETCH_HEAD')
    if ((git -C $Directory rev-parse HEAD) -ne $Spec.commit) { throw 'Source pin mismatch' }
    Invoke-Checked git @('-C',$Directory,'fsck','--full')
}
$vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
$vs = & $vswhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Preinstalled VS2022 C++ toolset missing' }
Import-Module "$vs/Common7/Tools/Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64 -vcvars_ver=14.44 -winsdk=10.0.26100.0'
if ($env:VCToolsVersion -notmatch '^14\.44\.' -or $env:WindowsSDKVersion -notmatch '^10\.0\.26100\.0') { throw 'Approved toolchain family unavailable' }
if ((cmake --version | Select-Object -First 1) -ne 'cmake version 3.31.6') { throw 'CMake pin mismatch' }
$ninja = (ninja --version).Trim()
if ([version]$ninja -lt [version]'1.12.1') { throw 'Ninja too old' }
$compiler = (& cl 2>&1 | Out-String)
$linker = (& link 2>&1 | Select-Object -First 5 | Out-String)
$toolchain = [ordered]@{
    image=$env:ImageOS; image_version=$env:ImageVersion; runner_os=$env:RUNNER_OS;
    compiler=$compiler; linker=$linker; msvc=$env:VCToolsVersion; windows_sdk=$env:WindowsSDKVersion;
    cmake=(cmake --version | Select-Object -First 1); ninja=$ninja;
    compatibility_review='VS2022 v143 / MSVC14.44 and SDK26100 retained; servicing versions recorded; Ninja >=1.12.1 is a build executor, not runtime arithmetic.'
}
if (($env:CL+' '+$env:_CL_) -match '(?i)[/-]Zc:cmath') { throw 'Incompatible injected math-header compiler option' }
Invoke-Checked cl @('/nologo','/std:c++17','/MD','/EHsc','/D_USE_MATH_DEFINES',"$tools/drivers/math_sanity.cpp",('/Fo'+$work+'/math-sanity.obj'),('/Fe'+$work+'/math-sanity.exe'))
Invoke-Checked "$work/math-sanity.exe" @()
$toolchain.math_sanity='PASS: command-line _USE_MATH_DEFINES precedes cmath; no /Zc:cmath in injected compiler options'
$metadata = Invoke-RestMethod -Uri 'https://vulkan.lunarg.com/sdk/files.json'
$sdkmeta = @($metadata.windows.($pins.sdk.version).files | Where-Object { $_.file_name -eq $pins.sdk.file })
if ($sdkmeta.Count -ne 1 -or $sdkmeta[0].sha -ne $pins.sdk.sha256 -or $sdkmeta[0].size -ne $pins.sdk.bytes) { throw 'Publisher SDK metadata differs from reviewed identity' }
$sdkmeta[0] | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$records/sdk-publisher.json" -Encoding utf8
$sdkexe = Join-Path $work $pins.sdk.file
$sdkurl = 'https://sdk.lunarg.com/sdk/download/'+$pins.sdk.version+'/windows/'+$pins.sdk.file
Invoke-WebRequest -Uri $sdkurl -OutFile $sdkexe
if ((Get-Item -LiteralPath $sdkexe).Length -ne $pins.sdk.bytes -or (Get-FileHash -LiteralPath $sdkexe -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pins.sdk.sha256) { throw 'SDK payload identity mismatch' }
$signature = Get-AuthenticodeSignature -LiteralPath $sdkexe
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'LunarG') { throw 'SDK publisher signature rejected' }
$sdkroot = Join-Path $work 'vulkan-sdk'
$sdkargs = @('--root',('"'+$sdkroot+'"'),'--accept-licenses','--default-answer','--confirm-command','install','copy_only=1')
$process = Start-Process -FilePath $sdkexe -ArgumentList $sdkargs -WindowStyle Hidden -Wait -PassThru
if ($process.ExitCode -ne 0) { throw "SDK copy-only extraction failed: $($process.ExitCode)" }
$env:VULKAN_SDK=$sdkroot
$env:Path="$sdkroot/Bin;"+$env:Path
if (-not (Test-Path -LiteralPath "$sdkroot/Include/vulkan/vulkan.hpp")) { throw 'SDK headers missing' }
$toolchain.sdk_sha256=$pins.sdk.sha256
$toolchain.sdk_signer=$signature.SignerCertificate.Subject
$toolchain.sdk_certificate_thumbprint=$signature.SignerCertificate.Thumbprint
$toolchain.sdk_exit=$process.ExitCode
$toolchain.shader_compiler=(& "$sdkroot/Bin/glslc.exe" --version | Out-String)
# The SDK provides import libraries, while the Windows loader is a separate
# same-version publisher component. CPU/help executables still import it.
$loadermeta=@($metadata.windows.($pins.runner_loader.version).files | Where-Object { $_.file_name -eq $pins.runner_loader.file })
if ($loadermeta.Count -ne 1 -or $loadermeta[0].sha -ne $pins.runner_loader.sha256 -or $loadermeta[0].size -ne $pins.runner_loader.bytes) { throw 'Runner loader publisher metadata differs' }
$loaderzip=Join-Path $work $pins.runner_loader.file
Invoke-WebRequest -Uri ('https://sdk.lunarg.com/sdk/download/'+$pins.runner_loader.version+'/windows/'+$pins.runner_loader.file) -OutFile $loaderzip
if ((Get-Item -LiteralPath $loaderzip).Length -ne $pins.runner_loader.bytes -or (Get-FileHash -LiteralPath $loaderzip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pins.runner_loader.sha256) { throw 'Runner loader payload identity mismatch' }
$loaderroot=Join-Path $work 'runner-loader'
Expand-Archive -LiteralPath $loaderzip -DestinationPath $loaderroot
$loaders=@(Get-ChildItem -LiteralPath $loaderroot -Filter vulkan-1.dll -Recurse -File | Where-Object {
    $pe=[IO.File]::ReadAllBytes($_.FullName);$peOffset=[BitConverter]::ToInt32($pe,60)
    [BitConverter]::ToUInt32($pe,$peOffset) -eq 0x4550 -and [BitConverter]::ToUInt16($pe,$peOffset+4) -eq 0x8664
})
if ($loaders.Count -ne 1) { throw 'Exactly one official x64 loader required' }
$loader=$loaders[0];$loaderSignature=Get-AuthenticodeSignature -LiteralPath $loader.FullName
if ($loaderSignature.Status -ne 'Valid' -or $loaderSignature.SignerCertificate.Subject -notmatch 'LunarG') { throw 'Runner loader signature rejected' }
$env:Path=$loader.DirectoryName+';'+$env:Path
Invoke-Checked python @('-c','import ctypes,sys; [ctypes.WinDLL(p) for p in sys.argv[1:]]; print("Runner DLL prerequisite load PASS")',$loader.FullName,'msvcp140.dll','vcruntime140.dll','vcruntime140_1.dll')
$toolchain.runner_loader=[ordered]@{archive_sha256=$pins.runner_loader.sha256;dll_sha256=(Get-FileHash -LiteralPath $loader.FullName -Algorithm SHA256).Hash.ToLowerInvariant();version=$loader.VersionInfo.FileVersion;signer=$loaderSignature.SignerCertificate.Subject;scope='process PATH only; not bundled; no Vulkan/model compute'}
$toolchain | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$records/toolchain.json" -Encoding utf8
Write-Host ($toolchain | ConvertTo-Json -Depth 8)
Write-Host ($sdkmeta[0] | ConvertTo-Json -Depth 8)
$native=Join-Path $work 'native'; $spirv=Join-Path $work 'spirv'; $webp=Join-Path $work 'webp'
Fetch-Pinned $pins.native $native
Fetch-Pinned $pins.ggml "$native/thirdparty/ggml"
Fetch-Pinned $pins.spirv $spirv
Fetch-Pinned $pins.webp $webp
$gitlink = (git -C $native ls-tree HEAD thirdparty/ggml)
if ($gitlink -notmatch $pins.ggml.commit) { throw 'Native submodule pin mismatch' }
foreach ($patch in $pins.patches.psobject.Properties) {
    $observed=(Get-FileHash -LiteralPath "$tools/patches/$($patch.Name)" -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "$($patch.Name): expected=$($patch.Value) actual=$observed"
    if ($observed -ne $patch.Value) { throw 'Reviewed patch hash mismatch' }
}
$prefix=Join-Path $work 'spirv-prefix'
Invoke-Checked cmake @('-S',$spirv,'-B',"$work/spirv-build",'-G','Ninja',"-DCMAKE_INSTALL_PREFIX=$prefix",'-DSPIRV_HEADERS_ENABLE_TESTS=OFF','-DSPIRV_HEADERS_ENABLE_INSTALL=ON')
Invoke-Checked cmake @('--install',"$work/spirv-build")
# Additive integration only; upstream algorithm sources remain unmodified in the control.
Add-Content -LiteralPath "$native/CMakeLists.txt" -Value ('`nadd_subdirectory("'+($tools.Replace('\','/'))+'/drivers" "${CMAKE_BINARY_DIR}/repair-drivers")').Replace('`n',"`n") -Encoding utf8
Copy-Item -LiteralPath "$tools/drivers/decoder_checkpoint.h" -Destination "$native/include/decoder_checkpoint.h"
$build=Join-Path $work 'build'
$options=@('-S',$native,'-B',$build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DBUILD_SHARED_LIBS=ON','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
    '-DCMAKE_C_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /Zi','-DCMAKE_CXX_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /Zi',
    '-DCMAKE_EXE_LINKER_FLAGS=/DEBUG:FULL /OPT:REF /OPT:ICF','-DCMAKE_SHARED_LINKER_FLAGS=/DEBUG:FULL /OPT:REF /OPT:ICF',
    '-DGGML_NATIVE=OFF','-DGGML_CPU=ON','-DGGML_VULKAN=ON','-DGGML_CUDA=OFF','-DGGML_HIP=OFF','-DGGML_METAL=OFF',
    '-DGGML_SYCL=OFF','-DGGML_WEBGPU=OFF','-DGGML_RPC=OFF','-DGGML_OPENMP=OFF','-DGGML_BUILD_TESTS=OFF','-DGGML_BUILD_EXAMPLES=OFF',
    '-DGGML_VULKAN_DEBUG=ON','-DPIXAL3D_GGML_PATCHES=OFF','-DTRELLIS_WEBP=ON',
    "-DFETCHCONTENT_SOURCE_DIR_LIBWEBP=$webp",'-DFETCHCONTENT_FULLY_DISCONNECTED=ON',
    "-DSPIRV-Headers_DIR=$prefix/share/cmake/SPIRV-Headers", "-DCMAKE_PREFIX_PATH=$prefix", "-DVulkan_INCLUDE_DIR=$sdkroot/Include",
    "-DVulkan_LIBRARY=$sdkroot/Lib/vulkan-1.lib","-DVulkan_GLSLC_EXECUTABLE=$sdkroot/Bin/glslc.exe")
$options | ConvertTo-Json | Set-Content -LiteralPath "$records/configure-options.json" -Encoding utf8
Invoke-Checked cmake $options
$commands=Get-Content -LiteralPath "$build/compile_commands.json" -Raw | ConvertFrom-Json
$nafCommands=@($commands | Where-Object { $_.file -match '[/\\]test_naf\.cpp$' })
if ($nafCommands.Count -ne 1 -or $nafCommands[0].command -notmatch '[/-]D_USE_MATH_DEFINES' -or $nafCommands[0].command -match '(?i)[/-]Zc:cmath') { throw 'NAF test compiler definition contract failed' }
$nafCommands[0] | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$records/naf-compile-command.json" -Encoding utf8
Write-Host ('Verified NAF compiler command: '+$nafCommands[0].command)
$targets=@('trellis-cli','trellis-server','repair-stage','repair-probe','repair-conditioning','repair-decoder','trellis-test-naf')
# Reuse the previously verified complete two-patch control; build only the new candidate.
foreach($patch in $pins.patches.psobject.Properties) {
    $target = if($patch.Name.StartsWith('ggml-')) { "$native/thirdparty/ggml" } else { $native }
    Invoke-Checked git @('-C',$target,'apply','--check',"$tools/patches/$($patch.Name)")
    Invoke-Checked git @('-C',$target,'apply',"$tools/patches/$($patch.Name)")
}
Invoke-Checked cmake (@('--build',$build,'--verbose','--parallel','4','--target')+$targets)
Invoke-Checked python @("$tools/scripts/package.py",'candidate',$work,$tools)
Invoke-Checked python @("$tools/scripts/package.py",'finish',$work,$tools)
Write-Host 'Complete candidate built and mandatory CPU/startup/geometry-writer checks passed. New local Vulkan/model gates NOT RUN on runner.'

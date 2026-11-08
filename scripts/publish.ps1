$ErrorActionPreference='Stop'
if ($env:GITHUB_ACTIONS -ne 'true' -or $env:GITHUB_REPOSITORY -ne 'swizzerul/pixal3d-vulkan-repair') { throw 'Authorized tools repository only' }
$files=@('candidate-windows-x64.zip','candidate-manifest.json','build-manifest.json','SHA256SUMS.txt')
$actual=@(Get-ChildItem -LiteralPath release -File | ForEach-Object Name)
if (Compare-Object ($files|Sort-Object) ($actual|Sort-Object)) { throw 'Unexpected release payload' }
foreach($f in $files){if((Get-Item -LiteralPath "release/$f").Length -ge 2147483648){throw 'Release asset limit'}}
$tag='experimental-'+$env:GITHUB_RUN_ID
$body="Experimental complete candidate: retained convolution/cache repairs plus existing chunked texture routing and an optional raw geometry checkpoint. Existing verified control is experimental-42506628576 and is not rebuilt. Build run: https://github.com/$env:GITHUB_REPOSITORY/actions/runs/$env:GITHUB_RUN_ID . CPU/startup checks only; new local Vulkan and model gates remain unverified. See build-manifest.json and SHA256SUMS.txt."
$body | Set-Content -LiteralPath "$env:RUNNER_TEMP/release-notes.txt" -Encoding utf8
gh release create $tag --repo $env:GITHUB_REPOSITORY --target $env:GITHUB_SHA --prerelease --title "Experimental Vulkan repair $env:GITHUB_RUN_ID" --notes-file "$env:RUNNER_TEMP/release-notes.txt" @($files|ForEach-Object{"release/$_"})
if($LASTEXITCODE){throw "Release publication failed: $LASTEXITCODE"}

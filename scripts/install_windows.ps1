param([switch]$NoLaunch, [switch]$NoShortcuts)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
    throw 'This installer supports Windows x64. Native ARM64 is not supported.'
}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$runtime = Join-Path $taskRoot '.runtime'
$cache = Join-Path $runtime 'downloads'
$pythonDir = Join-Path $runtime 'python'
New-Item -ItemType Directory -Force -Path $cache, $pythonDir | Out-Null
$config = Get-Content -LiteralPath (Join-Path $taskRoot 'config\windows-runtime.json') -Raw | ConvertFrom-Json

function Get-VerifiedDownload($Url, $Destination, $Hash) {
    if ((Test-Path -LiteralPath $Destination) -and (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -eq $Hash) { return }
    Write-Host "Downloading $([IO.Path]::GetFileName($Destination))..."
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Destination -TimeoutSec 900
    if ((Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -ne $Hash) {
        throw "Download checksum mismatch: $Destination. Run setup again."
    }
}

Write-Host 'Senyas setup - Windows x64'
Write-Host 'The first setup needs internet access. Python and libraries stay inside this folder.'
$pythonZip = Join-Path $cache 'python.zip'
Get-VerifiedDownload $config.python.url $pythonZip $config.python.sha256
if (-not (Test-Path -LiteralPath (Join-Path $pythonDir 'python.exe'))) {
    Expand-Archive -LiteralPath $pythonZip -DestinationPath $pythonDir -Force
}
# Explicit paths keep the embedded runtime isolated from global Python installs.
@('python311.zip', '.', 'Lib\site-packages', '..\..', 'import site') | Set-Content -LiteralPath (Join-Path $pythonDir 'python311._pth') -Encoding ASCII
$python = Join-Path $pythonDir 'python.exe'
$pipWheel = Join-Path $cache $config.pip.filename
Get-VerifiedDownload $config.pip.url $pipWheel $config.pip.sha256
& $python -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' $pipWheel (Join-Path $pythonDir 'Lib\site-packages')
if ($LASTEXITCODE -ne 0) { throw 'Could not install the pinned package installer.' }
Write-Host 'Installing pinned libraries. This can take several minutes...'
& $python -m pip --isolated install --disable-pip-version-check --only-binary=:all: --require-hashes --no-warn-script-location --timeout 180 -r (Join-Path $taskRoot 'requirements-windows.lock')
if ($LASTEXITCODE -ne 0) { throw 'Library installation failed. Check internet access, then run setup again.' }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Library compatibility check failed.' }
& $python (Join-Path $PSScriptRoot 'verify_install.py')
if ($LASTEXITCODE -ne 0) { throw 'The installed app did not pass its startup check.' }
if (-not $NoShortcuts) {
    $shell = New-Object -ComObject WScript.Shell
    foreach ($folder in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
        $link = $shell.CreateShortcut((Join-Path $folder 'Senyas.lnk'))
        $link.TargetPath = Join-Path $pythonDir 'pythonw.exe'
        $link.Arguments = '"' + (Join-Path $PSScriptRoot 'launch_desktop.py') + '"'
        $link.WorkingDirectory = $taskRoot
        $link.Description = 'Learn Filipino Sign Language with Senyas'
        $link.Save()
    }
}
Write-Host 'Senyas is ready. Use the shortcut or Start Senyas.cmd next time.'
if (-not $NoLaunch) {
    Start-Process -FilePath (Join-Path $pythonDir 'pythonw.exe') -ArgumentList ('"' + (Join-Path $PSScriptRoot 'launch_desktop.py') + '"') -WorkingDirectory $taskRoot -WindowStyle Hidden
}

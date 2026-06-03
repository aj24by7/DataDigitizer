$ErrorActionPreference = "Stop"

$sourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$targetDir = Join-Path $env:LOCALAPPDATA "DataDigitizer\2.11"
$desktopDir = [Environment]::GetFolderPath("Desktop")
$programsDir = [Environment]::GetFolderPath("Programs")
$startMenuDir = Join-Path $programsDir "Data Digitizer"
$downloadDir = Join-Path $env:TEMP "DataDigitizer-2.11-Installer"
$releaseBaseUrl = "https://github.com/aj24by7/DataDigitizer/releases/download/v2.11"

$apps = @(
    @{ File = "digitizer.exe"; Shortcut = "Digitizer.lnk"; Description = "Data Digitizer" },
    @{ File = "accuracytester.exe"; Shortcut = "AccuracyTester.lnk"; Description = "Accuracy Tester" },
    @{ File = "DataDigitizer-2.11.exe"; Shortcut = "DataDigitizer CLI.lnk"; Description = "Data Digitizer CLI" }
)

function Test-RealExe {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }

    $item = Get-Item -LiteralPath $Path
    if ($item.Length -lt 1048576) {
        return $false
    }

    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $b1 = $stream.ReadByte()
        $b2 = $stream.ReadByte()
        return ($b1 -eq 0x4D -and $b2 -eq 0x5A)
    }
    finally {
        $stream.Dispose()
    }
}

function Get-CandidateDirs {
    $dirs = @(
        $sourceDir,
        (Join-Path $sourceDir "dist"),
        (Join-Path $sourceDir "..\dist"),
        (Join-Path $sourceDir ".."),
        (Join-Path $sourceDir "..\..\2.11\dist")
    )

    $resolved = New-Object System.Collections.Generic.List[string]
    foreach ($dir in $dirs) {
        try {
            $path = (Resolve-Path -LiteralPath $dir -ErrorAction Stop).Path
            if (-not $resolved.Contains($path)) {
                $resolved.Add($path)
            }
        }
        catch {
        }
    }
    return $resolved
}

function Find-AppExe {
    param([string]$FileName)

    foreach ($dir in Get-CandidateDirs) {
        $candidate = Join-Path $dir $FileName
        if (Test-RealExe -Path $candidate) {
            return $candidate
        }
    }
    return $null
}

function Download-AppExe {
    param([string]$FileName)

    New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
    $destination = Join-Path $downloadDir $FileName
    if (Test-RealExe -Path $destination) {
        return $destination
    }

    $url = "$releaseBaseUrl/$FileName"
    Write-Host "Downloading $FileName from GitHub release..."
    Write-Host "  $url"

    try {
        Invoke-WebRequest -Uri $url -OutFile $destination -UseBasicParsing
    }
    catch {
        throw "Could not download $FileName. Connect to the internet, or download DataDigitizer-2.11-Windows11.zip from $releaseBaseUrl and run this installer from the extracted folder."
    }

    if (-not (Test-RealExe -Path $destination)) {
        throw "Downloaded $FileName, but it does not look like a valid Windows exe. Delete $downloadDir and run the installer again."
    }

    return $destination
}

New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
New-Item -ItemType Directory -Force -Path $startMenuDir | Out-Null

foreach ($app in $apps) {
    $source = Find-AppExe -FileName $app.File
    if ($null -eq $source) {
        Write-Host "$($app.File) was not found beside the installer."
        Write-Host "This usually means you downloaded GitHub's source-code ZIP instead of the Windows release ZIP."
        $source = Download-AppExe -FileName $app.File
    }
    Write-Host "Installing $($app.File)..."
    Copy-Item -LiteralPath $source -Destination (Join-Path $targetDir $app.File) -Force
}

$shell = New-Object -ComObject WScript.Shell

function New-Shortcut {
    param(
        [string]$ShortcutPath,
        [string]$TargetPath,
        [string]$Description
    )

    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath = $TargetPath
    $shortcut.WorkingDirectory = Split-Path -Parent $TargetPath
    $shortcut.IconLocation = $TargetPath
    $shortcut.Description = $Description
    $shortcut.Save()
}

foreach ($app in $apps) {
    $target = Join-Path $targetDir $app.File
    New-Shortcut -ShortcutPath (Join-Path $desktopDir $app.Shortcut) -TargetPath $target -Description $app.Description
    New-Shortcut -ShortcutPath (Join-Path $startMenuDir $app.Shortcut) -TargetPath $target -Description $app.Description
}

Write-Host "Data Digitizer 2.11 installed successfully."
Write-Host ""
Write-Host "Desktop shortcuts created:"
Write-Host "  Digitizer"
Write-Host "  AccuracyTester"
Write-Host "  DataDigitizer CLI"
Write-Host ""
Write-Host "Installed files:"
Write-Host "  $targetDir"
Write-Host ""
Write-Host "To pin Digitizer to the taskbar:"
Write-Host "  1. Double-click the Digitizer desktop shortcut."
Write-Host "  2. Right-click the Digitizer icon on the taskbar."
Write-Host "  3. Click Pin to taskbar."

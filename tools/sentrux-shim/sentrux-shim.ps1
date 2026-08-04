#requires -Version 7.2

[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RemainingArgs = @($RemainingArgs)

# ---- Pro activation (kept from canonical shim) ----

$Features = @(
    "dsm_export", "file_detail_panel", "evolution_details",
    "what_if_analysis", "agent_mcp", "rule_gates", "nine_color_modes"
)

function Get-LicensePath {
    if (-not [string]::IsNullOrWhiteSpace($env:SENTRUX_LICENSE_FILE)) {
        return $env:SENTRUX_LICENSE_FILE
    }
    $homeDir = [Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)
    if ([string]::IsNullOrWhiteSpace($homeDir)) { $homeDir = $HOME }
    if ($IsWindows) {
        $base = [Environment]::GetFolderPath([Environment+SpecialFolder]::ApplicationData)
        if ([string]::IsNullOrWhiteSpace($base)) { $base = $homeDir }
        return (Join-Path (Join-Path $base "sentrux") "license.json")
    }
    if ($IsMacOS) {
        return (Join-Path (Join-Path (Join-Path $homeDir "Library") "Application Support") (Join-Path "sentrux" "license.json"))
    }
    $configBase = if (-not [string]::IsNullOrWhiteSpace($env:XDG_CONFIG_HOME)) {
        $env:XDG_CONFIG_HOME
    } else {
        Join-Path $homeDir ".config"
    }
    return (Join-Path (Join-Path $configBase "sentrux") "license.json")
}

function Get-AutoDisabledPath {
    $licensePath = Get-LicensePath
    $licenseDir = Split-Path -Parent $licensePath
    return (Join-Path $licenseDir "auto-pro.disabled")
}

function Write-License {
    param([string]$Key, [string]$Source)
    if ([string]::IsNullOrWhiteSpace($Key)) { throw "license key cannot be empty" }
    $licensePath = Get-LicensePath
    $licenseDir = Split-Path -Parent $licensePath
    New-Item -ItemType Directory -Force -Path $licenseDir | Out-Null
    $preview = if ($Key.Length -le 8) { "********" } else { "{0}...{1}" -f $Key.Substring(0, 4), $Key.Substring($Key.Length - 4) }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Key)
        $hash = ($sha.ComputeHash($bytes) | Select-Object -First 8 | ForEach-Object { $_.ToString("x2") }) -join ""
    } finally { $sha.Dispose() }
    $license = [ordered]@{
        tier = "pro"; status = "active"; source = $Source
        key_preview = $preview; key_fingerprint = $hash
        activated_at = (Get-Date).ToUniversalTime().ToString("o")
        features = $Features
    }
    $license | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $licensePath -Encoding UTF8
}

function Clear-AutoDisabled {
    $path = Get-AutoDisabledPath
    if (Test-Path -LiteralPath $path -PathType Leaf) { Remove-Item -LiteralPath $path -Force }
}

function Ensure-AutoActivation {
    if ($env:SENTRUX_AUTO_PRO -notin @("1", "true", "True", "TRUE")) { return }
    $licensePath = Get-LicensePath
    $disabledPath = Get-AutoDisabledPath
    if ((Test-Path -LiteralPath $licensePath -PathType Leaf) -or (Test-Path -LiteralPath $disabledPath -PathType Leaf)) { return }
    Write-License "OSS-AUTO-PRO" "auto-open-source"
}

function Show-ProStatus {
    Ensure-AutoActivation
    $licensePath = Get-LicensePath
    if (-not (Test-Path -LiteralPath $licensePath -PathType Leaf)) {
        Write-Output "Tier: free"; Write-Output "Status: inactive"
        Write-Output "License: $licensePath"
        Write-Output "Features: check, gate, scan, mcp, plugin, analytics"
        return
    }
    $license = Get-Content -LiteralPath $licensePath -Raw | ConvertFrom-Json
    if ($license.tier -eq "pro" -and $license.status -eq "active") {
        Write-Output "Tier: pro"; Write-Output "Status: active"
        Write-Output "License: $licensePath"
        if ($license.PSObject.Properties["key_preview"]) { Write-Output "Key: $($license.key_preview)" }
        Write-Output "Features: $($Features -join ', ')"
        return
    }
    Write-Output "Tier: free"; Write-Output "Status: inactive"
    Write-Output "License: $licensePath"
}

function Deactivate-Pro {
    $licensePath = Get-LicensePath
    if (Test-Path -LiteralPath $licensePath -PathType Leaf) { Remove-Item -LiteralPath $licensePath -Force }
    $disabledPath = Get-AutoDisabledPath
    $disabledDir = Split-Path -Parent $disabledPath
    New-Item -ItemType Directory -Force -Path $disabledDir | Out-Null
    "disabled" | Set-Content -LiteralPath $disabledPath -Encoding UTF8
    Write-Output "Sentrux Pro deactivated"; Write-Output "Tier: free"
    Write-Output "Auto activation: disabled until sentrux pro activate <key>"
}

function Show-ProHelp {
    Write-Output "Manage local open-source Sentrux Pro activation"
    Write-Output ""; Write-Output "Usage: sentrux pro <COMMAND>"; Write-Output ""
    Write-Output "Commands:"
    Write-Output "  activate <key>  Save license and enable Pro features"
    Write-Output "  status          Show tier, status, license path, and features"
    Write-Output "  deactivate      Remove local license and return to free tier"
}

# ---- Pro dispatch ----

if ($RemainingArgs.Count -gt 0 -and $RemainingArgs[0] -eq "pro") {
    $proArgs = @($RemainingArgs | Select-Object -Skip 1)
    if ($proArgs.Count -eq 0 -or $proArgs[0] -in @("-h", "--help", "help")) {
        Show-ProHelp; exit 0
    }
    switch ($proArgs[0]) {
        "activate" {
            if ($proArgs.Count -lt 2) { throw "missing license key: sentrux pro activate <key>" }
            Clear-AutoDisabled; Write-License $proArgs[1] "local-open-source"
            Write-Output "Sentrux Pro activated"; Show-ProStatus; exit 0
        }
        "status" { Show-ProStatus; exit 0 }
        "deactivate" { Deactivate-Pro; exit 0 }
        default { throw "unknown pro command '$($proArgs[0])'. Try: sentrux pro --help" }
    }
}

# ---- Core resolution and execution ----

$script:ShimDir = Split-Path -Parent $PSCommandPath

function Resolve-Core {
    if (-not [string]::IsNullOrWhiteSpace($env:SENTRUX_CORE_EXE) -and (Test-Path -LiteralPath $env:SENTRUX_CORE_EXE -PathType Leaf)) {
        return (Get-Item -LiteralPath $env:SENTRUX_CORE_EXE).FullName
    }
    $separator = [System.IO.Path]::PathSeparator
    $pathEntries = @($env:PATH -split [regex]::Escape([string]$separator) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    foreach ($entry in $pathEntries) {
        foreach ($name in @("sentrux.exe", "sentrux-core.exe", "sentrux", "sentrux-core")) {
            $candidate = Join-Path $entry $name
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                $full = (Get-Item -LiteralPath $candidate).FullName
                if ($full -ne $PSCommandPath) { return $full }
            }
        }
    }
    throw "Sentrux core executable not found."
}

function Invoke-Core {
    param([string[]]$CoreArgs)
    Ensure-AutoActivation
    try {
        $core = Resolve-Core
        & $core @CoreArgs
    } catch {
        $liteCore = Join-Path $script:ShimDir "sentrux-lite-core.ps1"
        if (-not (Test-Path -LiteralPath $liteCore -PathType Leaf)) {
            throw "Sentrux core executable not found and lite core is missing."
        }
        & $liteCore @CoreArgs
    }
    exit $LASTEXITCODE
}

if ($RemainingArgs.Count -eq 0 -or $RemainingArgs[0] -in @("-h", "--help", "help")) {
    try {
        $core = Resolve-Core
        & $core --help 2>&1 | Out-String | Write-Output
        exit 0
    } catch {
        $liteCore = Join-Path $script:ShimDir "sentrux-lite-core.ps1"
        if (Test-Path -LiteralPath $liteCore -PathType Leaf) {
            & $liteCore --help 2>&1 | Out-String | Write-Output
        } else {
            Write-Output "Live codebase visualization and structural quality gate"
            Write-Output ""; Write-Output "Commands:"
            Write-Output "  pro        Manage local open-source Pro activation"
        }
        exit 0
    }
}

Invoke-Core -CoreArgs $RemainingArgs

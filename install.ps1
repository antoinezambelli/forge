# Forge Proxy bootstrap installer for Windows x64.
# One line: irm https://raw.githubusercontent.com/antoinezambelli/forge/main/install.ps1 | iex
# Save, inspect, execute:
#   iwr https://raw.githubusercontent.com/antoinezambelli/forge/main/install.ps1 -OutFile install.ps1
#   Get-Content .\install.ps1
#   .\install.ps1 -Version X.Y.Z -NoInit -InstallRoot 'C:\Forge Proxy'
# Manual immutable install:
#   iwr https://github.com/antoinezambelli/forge/releases/download/vX.Y.Z/proxy-X.Y.Z.json -OutFile proxy-X.Y.Z.json
#   # Download the windows-x86_64 artifact named by the manifest, then compare its size and:
#   Get-FileHash .\forge-proxy-windows-x86_64.exe -Algorithm SHA256
#   .\forge-proxy-windows-x86_64.exe install-artifact --version X.Y.Z --sha256 HEX

$ErrorActionPreference = "Stop"

function Show-Help {
    @'
Usage: install.ps1 [-Version X.Y.Z] [-NoInit] [-InstallRoot ABSOLUTE] [-Help]

Downloads, verifies, and hands a Windows x64 release to its install-artifact
operation. With no version, the stable pointer is used. Installation never
prompts; -NoInit remains accepted for explicit automation.

One line:
  irm https://raw.githubusercontent.com/antoinezambelli/forge/main/install.ps1 | iex

Save, inspect, execute:
  iwr https://raw.githubusercontent.com/antoinezambelli/forge/main/install.ps1 -OutFile install.ps1
  Get-Content .\install.ps1
  .\install.ps1 -Version X.Y.Z -NoInit -InstallRoot 'C:\Forge Proxy'

Manual immutable install:
  iwr https://github.com/antoinezambelli/forge/releases/download/vX.Y.Z/proxy-X.Y.Z.json -OutFile proxy-X.Y.Z.json
  # Download the windows-x86_64 artifact named by the manifest, then compare its size and:
  Get-FileHash .\forge-proxy-windows-x86_64.exe -Algorithm SHA256
  .\forge-proxy-windows-x86_64.exe install-artifact --version X.Y.Z --sha256 HEX
'@
}

function Test-Version([string]$Value) {
    return $Value -cmatch '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'
}

function Test-ExactProperties($Object, [string[]]$Expected) {
    if ($null -eq $Object -or $Object -isnot [pscustomobject]) { return $false }
    $names = @($Object.PSObject.Properties | ForEach-Object { $_.Name })
    if ($names.Count -ne $Expected.Count) { return $false }
    foreach ($name in $Expected) {
        if ($names -cnotcontains $name) { return $false }
    }
    return $true
}

function Get-NativeTarget([bool]$Testing) {
    if ($Testing) {
        $system = $env:_FORGE_PROXY_BOOTSTRAP_SYSTEM
        $machine = $env:_FORGE_PROXY_BOOTSTRAP_MACHINE
    } else {
        $system = if ($env:OS -eq 'Windows_NT') { 'Windows' } else { 'Unsupported' }
        $machine = if ($env:PROCESSOR_ARCHITEW6432) {
            $env:PROCESSOR_ARCHITEW6432
        } else {
            $env:PROCESSOR_ARCHITECTURE
        }
    }
    if ($system -ceq 'Windows' -and $machine -in @('AMD64', 'x86_64')) {
        return 'windows-x86_64'
    }
    throw "unsupported standalone target: $system $machine"
}

$version = $null
$noInit = $false
$installRoot = $null
$help = $false
$seen = @{}
for ($index = 0; $index -lt $args.Count; $index++) {
    $argument = $args[$index]
    if (@('-Version', '-NoInit', '-InstallRoot', '-Help') -cnotcontains $argument) {
        throw "unknown argument: $argument"
    }
    if ($seen.ContainsKey($argument)) { throw "duplicate argument: $argument" }
    $seen[$argument] = $true
    switch -CaseSensitive ($argument) {
        '-Version' {
            $index++
            if ($index -ge $args.Count) { throw '-Version requires X.Y.Z' }
            $version = $args[$index]
        }
        '-NoInit' { $noInit = $true }
        '-InstallRoot' {
            $index++
            if ($index -ge $args.Count) { throw '-InstallRoot requires an absolute path' }
            $installRoot = $args[$index]
        }
        '-Help' { $help = $true }
    }
}

if ($help) {
    if ($args.Count -ne 1) { throw '-Help cannot be combined with other arguments' }
    Show-Help
    exit 0
}
if ($null -ne $version -and -not (Test-Version $version)) {
    throw "invalid Proxy version: '$version'; expected X.Y.Z"
}
if ($null -ne $installRoot -and $installRoot -cnotmatch '^(?:[A-Za-z]:[\\/]|\\\\[^\\]+\\[^\\]+)') {
    throw '-InstallRoot must be an absolute path'
}

$testing = $env:_FORGE_PROXY_BOOTSTRAP_TESTING -ceq '1'
$target = Get-NativeTarget $testing
$pointerUrl = 'https://raw.githubusercontent.com/antoinezambelli/forge/main/installer/proxy-stable.txt'
$releaseBase = 'https://github.com/antoinezambelli/forge/releases/download'
$temporaryBase = [IO.Path]::GetTempPath()
if ($testing) {
    if ($env:_FORGE_PROXY_BOOTSTRAP_POINTER_URL) { $pointerUrl = $env:_FORGE_PROXY_BOOTSTRAP_POINTER_URL }
    if ($env:_FORGE_PROXY_BOOTSTRAP_RELEASE_BASE_URL) { $releaseBase = $env:_FORGE_PROXY_BOOTSTRAP_RELEASE_BASE_URL.TrimEnd('/') }
    if ($env:_FORGE_PROXY_BOOTSTRAP_TEMP_ROOT) { $temporaryBase = $env:_FORGE_PROXY_BOOTSTRAP_TEMP_ROOT }
}

$temporary = Join-Path $temporaryBase ("forge-proxy-bootstrap-" + [guid]::NewGuid().ToString('N'))
$exitCode = 1
try {
    [void](New-Item -ItemType Directory -Path $temporary)
    if ($null -eq $version) {
        $pointerPath = Join-Path $temporary 'proxy-stable.txt'
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $pointerUrl -OutFile $pointerPath
        } catch {
            $response = $_.Exception.Response
            $status = if ($null -ne $response) { [int]$response.StatusCode } else { $null }
            if ($status -eq 404) {
                throw 'no stable standalone Proxy release has been published'
            }
            throw "download unavailable: $pointerUrl"
        }
        $pointer = [IO.File]::ReadAllText($pointerPath, [Text.Encoding]::ASCII)
        if ($pointer.EndsWith("`n")) { $pointer = $pointer.Substring(0, $pointer.Length - 1) }
        if (-not (Test-Version $pointer)) { throw 'stable pointer must contain one bare X.Y.Z line' }
        $version = $pointer
    }

    $manifestUrl = "$releaseBase/v$version/proxy-$version.json"
    $manifestPath = Join-Path $temporary "proxy-$version.json"
    Invoke-WebRequest -UseBasicParsing -Uri $manifestUrl -OutFile $manifestPath
    try {
        $manifest = [IO.File]::ReadAllText($manifestPath) | ConvertFrom-Json
    } catch {
        throw 'release manifest is not valid JSON'
    }
    if (-not (Test-ExactProperties $manifest @('version', 'artifacts'))) {
        throw 'release manifest must contain only version and artifacts'
    }
    if ($manifest.version -cne $version -or -not (Test-ExactProperties $manifest.artifacts @($manifest.artifacts.PSObject.Properties.Name))) {
        throw 'release manifest version does not match the requested version'
    }

    $selected = $null
    foreach ($property in $manifest.artifacts.PSObject.Properties) {
        if ($property.Name -notin @('windows-x86_64', 'linux-x86_64-gnu', 'macos-arm64')) {
            throw "unsupported release target: $($property.Name)"
        }
        $entry = $property.Value
        if (-not (Test-ExactProperties $entry @('name', 'sha256', 'size'))) {
            throw "invalid release manifest entry for $($property.Name)"
        }
        $safeName = $entry.name -is [string] -and $entry.name.Length -gt 0 -and
            [IO.Path]::GetFileName($entry.name) -ceq $entry.name -and
            $entry.name -notin @('.', '..') -and $entry.name -cnotmatch '[\\/]'
        $validSize = ($entry.size -is [int] -or $entry.size -is [long]) -and $entry.size -ge 0
        if (-not $safeName -or $entry.sha256 -isnot [string] -or
            $entry.sha256 -cnotmatch '^[0-9a-fA-F]{64}$' -or -not $validSize) {
            throw "invalid release manifest entry for $($property.Name)"
        }
        if ($property.Name -ceq $target) { $selected = $entry }
    }
    if ($null -eq $selected) { throw "release manifest has no artifact for $target" }

    $artifactPath = Join-Path $temporary $selected.name
    $artifactUrl = "$releaseBase/v$version/$($selected.name)"
    Invoke-WebRequest -UseBasicParsing -Uri $artifactUrl -OutFile $artifactPath
    if ((Get-Item -LiteralPath $artifactPath).Length -ne [long]$selected.size) {
        throw 'downloaded artifact size does not match release manifest'
    }
    $actual = (Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $expected = $selected.sha256.ToLowerInvariant()
    if ($actual -cne $expected) { throw 'downloaded artifact checksum mismatch' }

    $handoff = @('install-artifact', '--version', $version, '--sha256', $expected)
    if ($noInit) { $handoff += '--no-init' }
    if ($null -ne $installRoot) { $handoff += @('--install-root', $installRoot) }
    & $artifactPath @handoff
    $exitCode = $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine("forge-proxy bootstrap: $($_.Exception.Message)")
} finally {
    if (Test-Path -LiteralPath $temporary) {
        Remove-Item -Recurse -Force -LiteralPath $temporary
    }
}
exit $exitCode

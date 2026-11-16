param(
    [string]$LaptopAddress = $laptopIp,
    [string]$PiAddress = $piIp,
    [switch]$SetupOnly
)

$ErrorActionPreference = 'Stop'

function Assert-TailnetAddress([string]$Value) {
    $address = $null
    if (-not [Net.IPAddress]::TryParse($Value, [ref]$address)) { throw 'Invalid Tailscale IPv4.' }
    $bytes = $address.GetAddressBytes()
    if ($bytes.Length -ne 4 -or $bytes[0] -ne 100 -or $bytes[1] -lt 64 -or $bytes[1] -gt 127 -or $address.ToString() -ne $Value) {
        throw 'Use the actual 100.64.0.0/10 Tailscale IPv4, not localhost or a LAN address.'
    }
}

function Assert-PrivatePath([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Settings must not be a link or junction.' }
    $acl = Get-Acl -LiteralPath $Path
    $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User
    if ($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -ne $sid.Value) { throw 'Settings owner must be the current Windows user.' }
    foreach ($rule in $acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])) {
        if ($rule.AccessControlType -eq 'Allow' -and $rule.IdentityReference.Value -ne $sid.Value) {
            throw 'Settings permissions are too broad; stop and review them before starting.'
        }
    }
}

try {
    if (-not $env:LOCALAPPDATA) { throw 'Run this launcher in Windows PowerShell.' }
    $python = Join-Path $env:LOCALAPPDATA 'emotion-stt-test\.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'The laptop STT Python environment is missing. Follow README first.' }
    $directory = Join-Path $env:LOCALAPPDATA 'emotion-stt-test\connection'
    $path = Join-Path $directory 'receiver.json'
    if (Test-Path -LiteralPath $directory) { Assert-PrivatePath $directory }

    if (Test-Path -LiteralPath $path) {
        Assert-PrivatePath $path
        try {
            $settings = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
            $secure = ConvertTo-SecureString $settings.token_dpapi
            $token = [Net.NetworkCredential]::new('', $secure).Password
        } catch { throw 'Cannot read/decrypt saved settings. They were not overwritten; use the original Windows account.' }
        $LaptopAddress = [string]$settings.laptop_ip
        $PiAddress = [string]$settings.pi_ip
        if ($settings.port -ne 8765) { throw 'Unexpected saved port; settings were not changed.' }
        Write-Host '[start] Using saved settings; existing terminal values are not replacing them.'
    } else {
        if (-not $LaptopAddress) { $LaptopAddress = Read-Host 'Laptop Tailscale IPv4' }
        if (-not $PiAddress) { $PiAddress = Read-Host 'Pi Tailscale IPv4' }
        $token = $env:COMPANION_AUDIO_TOKEN
        if (-not $token) {
            $secure = Read-Host 'Paste the SAME audio token used on the Pi (hidden)' -AsSecureString
            $token = [Net.NetworkCredential]::new('', $secure).Password
        }
    }
    Assert-TailnetAddress $LaptopAddress
    Assert-TailnetAddress $PiAddress
    if ($LaptopAddress -eq $PiAddress) { throw 'Laptop and Pi addresses must be different.' }
    if ($token -cnotmatch '\A[0-9a-fA-F]{64}\z') { throw 'The shared token must contain exactly 64 hex characters; no token was printed or generated.' }

    if (-not (Test-Path -LiteralPath $path)) {
        $secure = ConvertTo-SecureString $token -AsPlainText -Force
        $settings = [ordered]@{ laptop_ip = $LaptopAddress; pi_ip = $PiAddress; port = 8765; token_dpapi = (ConvertFrom-SecureString $secure) }
        if (-not (Test-Path -LiteralPath $directory)) {
            [void][IO.Directory]::CreateDirectory($directory)
            $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User
            $acl = [Security.AccessControl.DirectorySecurity]::new()
            $acl.SetOwner($sid)
            $acl.SetAccessRuleProtection($true, $false)
            $rule = [Security.AccessControl.FileSystemAccessRule]::new($sid, 'FullControl', 'ContainerInherit, ObjectInherit', 'None', 'Allow')
            $acl.AddAccessRule($rule)
            Set-Acl -LiteralPath $directory -AclObject $acl
        }
        Assert-PrivatePath $directory
        $bytes = [Text.Encoding]::UTF8.GetBytes(($settings | ConvertTo-Json))
        $file = [IO.File]::Open($path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try { $file.Write($bytes, 0, $bytes.Length) } finally { $file.Dispose() }
        Assert-PrivatePath $path
        Write-Host "[start] Saved settings: $path (Windows DPAPI token; current-user access)."
    }
    Write-Host "[start] Laptop ${LaptopAddress}:8765; permitted Pi $PiAddress. Token is not displayed."
    if ($SetupOnly) { Write-Host '[start] Setup checked. Nothing was started.'; return }

    & $python -B -c 'import av, faster_whisper'
    if ($LASTEXITCODE -ne 0) { throw 'STT dependencies are missing; use the separate laptop requirements, not the Pi environment.' }
    $oldToken = $env:COMPANION_AUDIO_TOKEN
    $root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
    Push-Location -LiteralPath $root
    try {
        $env:COMPANION_AUDIO_TOKEN = $token
        Write-Host '[start] Waiting for Pi audio. Ctrl+C stops the receiver, NOT the Pi microphone.'
        & $python -B (Join-Path $PSScriptRoot 'receive_audio.py') --host $LaptopAddress --allow-peer $PiAddress --port 8765
        if ($LASTEXITCODE -ne 0) { throw 'Receiver stopped with an error. Check its message above; do not disable the firewall.' }
    } finally {
        $env:COMPANION_AUDIO_TOKEN = $oldToken
        Pop-Location
    }
} catch {
    Write-Host "[start] $($_.Exception.Message)" -ForegroundColor Red
    throw 'Launcher stopped. No microphone was opened by this launcher.'
}

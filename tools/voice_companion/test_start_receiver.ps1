# Isolated synthetic settings only; no real user token, model, microphone or server.
$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'start_receiver.ps1'
$originalLocal = $env:LOCALAPPDATA
$originalToken = $env:COMPANION_AUDIO_TOKEN
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('voice-launcher-test-' + [guid]::NewGuid().ToString('N'))
$fakeToken = 'a' * 64
$script:checks = 0

function Assert-Test($Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
    $script:checks++
}
function New-Fixture([string]$Name) {
    $env:LOCALAPPDATA = Join-Path $testRoot $Name
    $scripts = Join-Path $env:LOCALAPPDATA 'emotion-stt-test\.venv\Scripts'
    [void][IO.Directory]::CreateDirectory($scripts)
    [IO.File]::WriteAllBytes((Join-Path $scripts 'python.exe'), [byte[]]@())
    $env:COMPANION_AUDIO_TOKEN = $fakeToken
}
function Read-Host { throw 'Unexpected prompt during inherited/saved-settings test.' }

try {
    New-Fixture 'inherited'
    $laptopIp = '100.74.11.81'
    $piIp = '100.74.11.82'
    $output = (& $launcher -SetupOnly 6>&1 | Out-String)
    $path = Join-Path $env:LOCALAPPDATA 'emotion-stt-test\connection\receiver.json'
    Assert-Test (Test-Path -LiteralPath $path) 'Settings not saved.'
    $json = Get-Content -LiteralPath $path -Raw
    $settings = $json | ConvertFrom-Json
    Assert-Test ($settings.laptop_ip -eq $laptopIp -and $settings.pi_ip -eq $piIp) 'Caller IP variables not inherited.'
    Assert-Test (-not $json.Contains($fakeToken) -and -not $output.Contains($fakeToken)) 'Plain token exposed.'
    $decoded = [Net.NetworkCredential]::new('', (ConvertTo-SecureString $settings.token_dpapi)).Password
    Assert-Test ($decoded -ceq $fakeToken) 'DPAPI did not preserve the existing token.'
    $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $acl = Get-Acl -LiteralPath $path
    $otherRules = @($acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]) | Where-Object { $_.AccessControlType -eq 'Allow' -and $_.IdentityReference.Value -ne $sid })
    Assert-Test ($otherRules.Count -eq 0) 'Other identities received file access.'

    $env:COMPANION_AUDIO_TOKEN = 'b' * 64
    $laptopIp = $null
    $piIp = $null
    & $launcher -SetupOnly | Out-Null
    Assert-Test ((Get-Content -LiteralPath $path -Raw) -ceq $json) 'Saved configuration was replaced.'

    New-Fixture 'invalid'
    $failed = $false
    try { & $launcher -SetupOnly -LaptopAddress '192.168.1.1' -PiAddress '100.74.11.82' 6>$null } catch { $failed = $true }
    Assert-Test $failed 'LAN address was accepted.'
    Assert-Test (-not (Test-Path (Join-Path $env:LOCALAPPDATA 'emotion-stt-test\connection\receiver.json'))) 'Invalid settings were saved.'

    New-Fixture 'unsafe-permissions'
    [void][IO.Directory]::CreateDirectory((Join-Path $env:LOCALAPPDATA 'emotion-stt-test\connection'))
    $failed = $false
    try { & $launcher -SetupOnly -LaptopAddress '100.74.11.81' -PiAddress '100.74.11.82' 6>$null } catch { $failed = $true }
    Assert-Test $failed 'Broad inherited permissions were accepted.'

    New-Fixture 'bad-token'
    $env:COMPANION_AUDIO_TOKEN = 'synthetic-invalid-secret'
    $failed = $false
    try { & $launcher -SetupOnly -LaptopAddress '100.74.11.81' -PiAddress '100.74.11.82' 6>$null } catch { $failed = $true }
    Assert-Test $failed 'Invalid token accepted.'

    New-Fixture 'missing-values'
    $env:COMPANION_AUDIO_TOKEN = $null
    $promptLog = [Collections.Generic.List[string]]::new()
    function Read-Host {
        param([string]$Prompt, [switch]$AsSecureString)
        $promptLog.Add($Prompt)
        if ($AsSecureString) { return ConvertTo-SecureString $fakeToken -AsPlainText -Force }
        if ($Prompt -like 'Laptop*') { return '100.74.11.81' }
        if ($Prompt -like 'Pi*') { return '100.74.11.82' }
        throw 'Unexpected prompt.'
    }
    $output = (& $launcher -SetupOnly 6>&1 | Out-String)
    Assert-Test ($promptLog.Count -eq 3) 'Missing values did not prompt exactly once each.'
    Assert-Test (-not $output.Contains($fakeToken)) 'Prompted token appeared in output.'
    Write-Host "PASS: $script:checks launcher checks. Synthetic fixtures retained at $testRoot"
} finally {
    $env:LOCALAPPDATA = $originalLocal
    $env:COMPANION_AUDIO_TOKEN = $originalToken
}

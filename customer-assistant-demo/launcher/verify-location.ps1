# Verify PowerPoint-style relocation without starting Python or a browser.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$artifactsRoot = Split-Path -Parent $projectRoot
$launcherPath = Join-Path $artifactsRoot 'Launch Customer Assistant.exe'
$testRoot = Join-Path $projectRoot 'data\local\launcher-location-checks'
$copiedDirectory = Join-Path $testRoot 'PowerPoint temporary copy'
$portableDirectory = Join-Path $testRoot 'Portable copy'
$portableProject = Join-Path $portableDirectory 'customer-assistant-demo'
New-Item -ItemType Directory -Path $copiedDirectory, $portableProject -Force | Out-Null
Copy-Item -LiteralPath $launcherPath -Destination (Join-Path $copiedDirectory 'Launch Customer Assistant.exe') -Force
[System.IO.File]::WriteAllText((Join-Path $portableProject 'app.py'), '# Location-only fixture; never run.')

$assembly = [System.Reflection.Assembly]::LoadFile($launcherPath)
$program = $assembly.GetType('Program')
$flags = [System.Reflection.BindingFlags]::NonPublic -bor [System.Reflection.BindingFlags]::Static
$locate = $program.GetMethod('LocateProject', $flags)
if ($null -eq $locate) { throw 'The rebuilt launcher has no project locator.' }

$cases = @(
    @{Name='Normal adjacent project'; Directory=$artifactsRoot; Expected=$projectRoot},
    @{Name='PowerPoint temporary copy'; Directory=$copiedDirectory; Expected=$projectRoot},
    @{Name='Portable sibling takes precedence'; Directory=$portableDirectory; Expected=$portableProject}
)
foreach ($case in $cases) {
    $actual = $locate.Invoke($null, [object[]] @([string] $case.Directory))
    if ($actual -ne [System.IO.Path]::GetFullPath($case.Expected)) {
        throw ($case.Name + ': wrong project directory: ' + $actual)
    }
    Write-Output ('PASS: ' + $case.Name)
}
$copiedAssembly = [System.Reflection.Assembly]::LoadFile((Join-Path $copiedDirectory 'Launch Customer Assistant.exe'))
$copiedLocator = $copiedAssembly.GetType('Program').GetMethod('LocateProject', $flags)
if ($copiedLocator.Invoke($null, [object[]] @([string] $copiedDirectory)) -ne $projectRoot) {
    throw 'The actual executable copy lost its installed project path.'
}
Write-Output 'PASS: Actual executable copy retains installation path'

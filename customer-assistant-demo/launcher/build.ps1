# Build the small Windows launcher using the installed .NET Framework compiler.
# No packaging or Python dependencies are downloaded or bundled.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$artifactsRoot = Split-Path -Parent $projectRoot
$outputPath = Join-Path $artifactsRoot 'Launch Customer Assistant.exe'
$sourcePath = Join-Path $PSScriptRoot 'Launcher.cs'
$compilerPath = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compilerPath)) {
    $compilerPath = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe'
}
if (-not (Test-Path -LiteralPath $compilerPath)) {
    throw 'The Windows .NET Framework C# compiler is missing.'
}
$locationPath = Join-Path ([System.IO.Path]::GetTempPath()) ('CustomerAssistantLocation-' + [Guid]::NewGuid().ToString('N') + '.txt')
try {
    # An embedded PowerPoint copy can still find this installation. A demo
    # folder beside the executable takes precedence for portable copies.
    [System.IO.File]::WriteAllText($locationPath, $projectRoot)
    & $compilerPath /nologo /target:winexe /optimize+ /reference:System.Windows.Forms.dll /reference:System.Drawing.dll "/out:$outputPath" "/resource:$locationPath,CustomerAssistant.ProjectDirectory" $sourcePath
    if ($LASTEXITCODE -ne 0) { throw 'Launcher compilation failed.' }
}
finally {
    if (Test-Path -LiteralPath $locationPath) { Remove-Item -LiteralPath $locationPath }
}
Get-Item -LiteralPath $outputPath | Select-Object FullName, Length

# PowerShell script to set up C# MGBT wrapper
# Run this after installing .NET SDK 6.0

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "C# MGBT Wrapper Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check .NET SDK
Write-Host "Step 1: Checking .NET SDK..." -ForegroundColor Yellow
$dotnetVersion = dotnet --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ .NET SDK found: $dotnetVersion" -ForegroundColor Green
} else {
    Write-Host "  ✗ .NET SDK not found!" -ForegroundColor Red
    Write-Host "  Please install .NET 6.0 SDK from:" -ForegroundColor Red
    Write-Host "  https://dotnet.microsoft.com/download/dotnet/6.0" -ForegroundColor Red
    Write-Host ""
    Write-Host "  After installation:" -ForegroundColor Yellow
    Write-Host "  1. Close this terminal" -ForegroundColor Yellow
    Write-Host "  2. Open a new terminal" -ForegroundColor Yellow
    Write-Host "  3. Run this script again" -ForegroundColor Yellow
    exit 1
}

# Step 2: Check Numerics repository
Write-Host ""
Write-Host "Step 2: Checking Numerics repository..." -ForegroundColor Yellow
$numericsPath = "D:\GitRepos\Numerics"
if (Test-Path $numericsPath) {
    Write-Host "  ✓ Numerics repository found" -ForegroundColor Green
} else {
    Write-Host "  ✗ Numerics repository not found at $numericsPath" -ForegroundColor Red
    Write-Host "  It should have been cloned. Please check." -ForegroundColor Red
    exit 1
}

# Step 3: Build Numerics library
Write-Host ""
Write-Host "Step 3: Building Numerics library..." -ForegroundColor Yellow
Push-Location $numericsPath
try {
    dotnet build -c Release 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Numerics library built successfully" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Build failed" -ForegroundColor Red
        dotnet build -c Release
        exit 1
    }
} finally {
    Pop-Location
}

# Step 4: Find Numerics.dll
Write-Host ""
Write-Host "Step 4: Locating Numerics.dll..." -ForegroundColor Yellow
$numericsDll = Get-ChildItem -Path $numericsPath -Filter "Numerics.dll" -Recurse | 
               Where-Object { $_.FullName -like "*Release*" } | 
               Select-Object -First 1

if ($numericsDll) {
    Write-Host "  ✓ Found: $($numericsDll.FullName)" -ForegroundColor Green
} else {
    Write-Host "  ✗ Numerics.dll not found" -ForegroundColor Red
    exit 1
}

# Step 5: Create wrapper project
Write-Host ""
Write-Host "Step 5: Creating C# wrapper project..." -ForegroundColor Yellow
$wrapperDir = "D:\GitRepos\MGBT_1.0.7\scripts\csharp_wrapper"
if (!(Test-Path $wrapperDir)) {
    New-Item -ItemType Directory -Path $wrapperDir | Out-Null
}

# Create project file
$projectContent = @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
    <RootNamespace>MGBTWrapper</RootNamespace>
  </PropertyGroup>
  <ItemGroup>
    <Reference Include="Numerics">
      <HintPath>$($numericsDll.FullName)</HintPath>
    </Reference>
  </ItemGroup>
</Project>
"@

$projectContent | Out-File -FilePath "$wrapperDir\csharp_mgbt_wrapper.csproj" -Encoding UTF8
Write-Host "  ✓ Project file created" -ForegroundColor Green

# Copy wrapper source
Copy-Item "D:\GitRepos\MGBT_1.0.7\scripts\csharp_mgbt_wrapper.cs" -Destination "$wrapperDir\Program.cs"
Write-Host "  ✓ Source code copied" -ForegroundColor Green

# Step 6: Build wrapper
Write-Host ""
Write-Host "Step 6: Building C# wrapper..." -ForegroundColor Yellow
Push-Location $wrapperDir
try {
    dotnet build -c Release 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Wrapper built successfully" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Build failed" -ForegroundColor Red
        dotnet build -c Release
        exit 1
    }
} finally {
    Pop-Location
}

# Step 7: Copy executable and dependencies
Write-Host ""
Write-Host "Step 7: Copying files to scripts directory..." -ForegroundColor Yellow
$exePath = "$wrapperDir\bin\Release\net6.0\csharp_mgbt_wrapper.exe"
$targetDir = "D:\GitRepos\MGBT_1.0.7\scripts"

if (Test-Path $exePath) {
    Copy-Item $exePath -Destination $targetDir -Force
    Write-Host "  ✓ Copied csharp_mgbt_wrapper.exe" -ForegroundColor Green
    
    # Copy Numerics.dll
    Copy-Item $numericsDll.FullName -Destination $targetDir -Force
    Write-Host "  ✓ Copied Numerics.dll" -ForegroundColor Green
    
    # Copy other dependencies
    $binDir = "$wrapperDir\bin\Release\net6.0"
    Get-ChildItem -Path $binDir -Filter "*.dll" | ForEach-Object {
        Copy-Item $_.FullName -Destination $targetDir -Force
    }
    Write-Host "  ✓ Copied dependencies" -ForegroundColor Green
} else {
    Write-Host "  ✗ Executable not found at $exePath" -ForegroundColor Red
    exit 1
}

# Step 8: Test the wrapper
Write-Host ""
Write-Host "Step 8: Testing C# wrapper..." -ForegroundColor Yellow
Push-Location $targetDir
try {
    $testOutput = & ".\csharp_mgbt_wrapper.exe" 100 200 300 10 20 400 500 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Wrapper test successful!" -ForegroundColor Green
        Write-Host "  Output: $testOutput" -ForegroundColor Gray
    } else {
        Write-Host "  ✗ Wrapper test failed" -ForegroundColor Red
        Write-Host "  Error: $testOutput" -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}

# Success!
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "✓ C# MGBT Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run the four-way comparison:" -ForegroundColor Cyan
Write-Host "  python scripts\compare_all_methods.py" -ForegroundColor White
Write-Host ""
Write-Host "Or test C# directly:" -ForegroundColor Cyan
Write-Host "  cd scripts" -ForegroundColor White
Write-Host "  .\csharp_mgbt_wrapper.exe 100 200 300 10 20 400 500" -ForegroundColor White
Write-Host ""

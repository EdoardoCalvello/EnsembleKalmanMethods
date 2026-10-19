# Run full pipeline for EnsembleKalmanMethods on Windows (PowerShell)
$ErrorActionPreference = "Stop"

function Check-Python {
    try {
        python --version | Out-Null
        return $true
    } catch {
        Write-Error "Python not found in PATH. Install Python 3.8+ and re-run."
        exit 1
    }
}

if (-not (Check-Python)) { exit 1 }

if (-not (Test-Path .\.venv)) {
    Write-Host "Creating virtual environment .\.venv..."
    python -m venv .venv
}

$venvPython = Join-Path -Path (Resolve-Path .\.venv).Path -ChildPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Cannot find $venvPython. Virtual environment creation failed."
    exit 1
}

Write-Host "Upgrading pip and installing requirements..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r .\requirements.txt

# Ensure results folders exist
New-Item -ItemType Directory -Force -Path .\Filtering_Experiments\results | Out-Null
New-Item -ItemType Directory -Force -Path .\Inversion_Experiments\results | Out-Null
New-Item -ItemType Directory -Force -Path .\Lorenz96\results | Out-Null

# Make sure Python can import modules from repository root
$repoRoot = (Get-Location).Path
$env:PYTHONPATH = $repoRoot

Write-Host "Running Lorenz96 data generation (in Lorenz96 folder)..."
Push-Location .\Lorenz96
& $venvPython .\data_gen.py
Pop-Location

Write-Host "Running filtering experiments..."
Push-Location .\Filtering_Experiments
Write-Host "Running filtering experiments (in Filtering_Experiments folder)..."
& $venvPython .\3DVAR_L96m.py
& $venvPython .\3DVAR_L96s.py
& $venvPython .\EnKF_L96s.py

Write-Host "Generating filtering plots (in Filtering_Experiments folder)..."
& $venvPython .\generate_plots.py
Pop-Location

Write-Host "Running inversion experiments..."
Push-Location .\Inversion_Experiments
Write-Host "Running inversion experiments (in Inversion_Experiments folder)..."
& $venvPython .\EKI_1D.py
& $venvPython .\EKI_L96s.py

Write-Host "Generating inversion plots (in Inversion_Experiments folder)..."
& $venvPython .\generate_plots.py
Pop-Location

Write-Host "Pipeline finished. Check './Filtering_Experiments/results' and './Inversion_Experiments/results' for outputs."

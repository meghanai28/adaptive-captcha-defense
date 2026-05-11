# Full evaluation + training pipeline.  Run from src/ directory: .\run_pipeline.ps1
# Order: native/cross/held-out/augtest evals -> baselines -> sensitivity -> train ablations -> eval ablations

$ErrorActionPreference = "Continue"
$start = Get-Date

function Step($n, $total, $name, $script) {
    Write-Host ""
    Write-Host "[$n/$total] $name" -ForegroundColor Magenta
    Write-Host "    Started: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    & $script
    Write-Host "    Finished: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
}

Step 1 5 "Main evals (native + cross-env + held-out + augtest)" { .\eval_all.ps1 }
Step 2 5 "Baseline comparison"                                   { .\eval_baselines.ps1 }
Step 3 5 "Sensitivity analysis"                                  { .\eval_sensitivity.ps1 }
Step 4 5 "Ablation training (40 runs)"                          { .\train_ablations.ps1 }
Step 5 5 "Ablation evaluation"                                   { .\eval_ablations.ps1 }

$elapsed = (Get-Date) - $start
Write-Host ""
Write-Host "Pipeline complete in $([int]$elapsed.TotalMinutes) min." -ForegroundColor Green

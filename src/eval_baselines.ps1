# Evaluate rule-based baselines.
# Run from src/ directory:  .\eval_baselines.ps1

$python = ".\venv\Scripts\python"

New-Item -ItemType Directory -Force -Path logs | Out-Null

function Run-Baselines($preset) {
    $log = "logs/baselines_${preset}.log"
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Evaluation complete" -Quiet)) {
        Write-Host "  Skipping baselines_${preset} - already done" -ForegroundColor Yellow
        return
    }
    Write-Host "  Running baselines in ${preset} environment..." -ForegroundColor Cyan
    & $python -u -m rl_captcha.scripts.evaluate_baselines `
        --data-dir "data/" `
        --reward-preset $preset `
        --episodes 500 `
        --split test `
        --split-seed 42 `
        --eval-seeds 42 123 456 789 1024 | Tee-Object -FilePath $log
    Write-Host "  Done -> $log" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== BASELINES: v2 environment (primary) ===" -ForegroundColor Magenta
Run-Baselines "v2"

Write-Host ""
Write-Host "=== BASELINES: v1 environment (legacy comparison) ===" -ForegroundColor Magenta
Run-Baselines "v1"

Write-Host ""
Write-Host "All baseline evaluations complete." -ForegroundColor Green

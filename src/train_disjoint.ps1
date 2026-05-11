# Disjoint generalization: train with Tier 5 (LLM) bots completely excluded,
# then evaluate only on Tier 5 sessions.  Tests whether the temporal agent
# generalises to bot families it has never seen during training.
#
# Run from src/ directory:  .\train_disjoint.ps1

$python  = ".\venv\Scripts\python"
$ckptDir = "rl_captcha/agent/checkpoints"
$seeds   = @(42, 123, 456, 789, 1024)
$preset  = "v2"
$aug     = "advaug"
$name_tag = "disjoint_tier5"

New-Item -ItemType Directory -Force -Path logs | Out-Null

$total = $seeds.Count
$run   = 0

Write-Host ""
Write-Host "=== DISJOINT TRAINING: Tier 5 (LLM) held out, $total seeds ===" -ForegroundColor Magenta
Write-Host ""

foreach ($seed in $seeds) {
    $run++
    $name = "ppo_${aug}_${preset}_${name_tag}_seed${seed}"
    $ckpt = "$ckptDir/$name"
    $log  = "logs/${name}_training.log"

    Write-Host "[$run/$total] $name" -ForegroundColor Cyan

    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Training complete\." -Quiet)) {
        Write-Host "  Skipping - already fully trained" -ForegroundColor Yellow
        continue
    }

    & $python -u -m rl_captcha.scripts.train_ppo `
        --algorithm       ppo `
        --reward-preset   $preset `
        --adversarial-augment `
        --held-out-tiers  5 `
        --train-seed      $seed `
        --data-dir        "data/" `
        --save-path       $ckpt `
        --total-timesteps 500000 | Tee-Object -FilePath $log

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED - check $log" -ForegroundColor Red
    } else {
        Write-Host "  Done -> $ckpt" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "All $total disjoint training runs complete." -ForegroundColor Green
Write-Host "Next: run .\eval_disjoint.ps1" -ForegroundColor Cyan

# Evaluate disjoint-trained models on Tier 5 (LLM) test sessions only.
# Compares against baseline (trained WITH Tier 5) to show how well the
# temporal agent generalises to never-seen bot families.
#
# Run from src/ directory:  .\eval_disjoint.ps1

$python  = ".\venv\Scripts\python"
$ckptDir = "rl_captcha/agent/checkpoints"
$seeds   = @(42, 123, 456, 789, 1024)
$preset  = "v2"
$aug     = "advaug"
$log     = "logs/eval_disjoint_tier5.log"

New-Item -ItemType Directory -Force -Path logs | Out-Null

if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Evaluation complete" -Quiet)) {
    Write-Host "Already done -> $log" -ForegroundColor Yellow
    exit 0
}

# Baseline agents (trained WITH Tier 5)
$baselineArgs = @()
foreach ($seed in $seeds) {
    $name = "ppo_${aug}_${preset}_seed${seed}"
    $baselineArgs += "${name}=${ckptDir}/${name}"
}

# Disjoint agents (trained WITHOUT Tier 5)
$disjointArgs = @()
foreach ($seed in $seeds) {
    $name = "ppo_${aug}_${preset}_disjoint_tier5_seed${seed}"
    $disjointArgs += "${name}=${ckptDir}/${name}"
}

Write-Host ""
Write-Host "=== DISJOINT EVAL: Tier 5 (LLM) held-out test set ===" -ForegroundColor Magenta
Write-Host ""

& $python -u -m rl_captcha.scripts.evaluate_ppo `
    --agent @($baselineArgs + $disjointArgs) `
    --data-dir    "data/" `
    --reward-preset $preset `
    --episodes    500 `
    --split       test `
    --split-seed  42 `
    --held-out-tiers 5 `
    --eval-seeds  42 123 456 789 1024 | Tee-Object -FilePath $log

Write-Host ""
Write-Host "Done -> $log" -ForegroundColor Green

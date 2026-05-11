# Re-evaluate key agents in strict mode: human_passed_puzzle = false positive.
# Produces separate _strict.log files — original logs are untouched.
# Run from src/ directory:  .\eval_strict.ps1

$python  = ".\venv\Scripts\python"
$ckptDir = "rl_captcha/agent/checkpoints"
$seeds   = @(42, 123, 456, 789, 1024)
$preset  = "v2"
$aug     = "advaug"
$algos   = @("ppo", "dg", "soft_ppo")

New-Item -ItemType Directory -Force -Path logs | Out-Null

# Build agent args for all three algorithms (for baseline strict eval)
$allBaselineArgs = @()
foreach ($algo in $algos) {
    foreach ($seed in $seeds) {
        $name = "${algo}_${aug}_${preset}_seed${seed}"
        $allBaselineArgs += "${name}=${ckptDir}/${name}"
    }
}

# PPO-only baseline args (used alongside ablation agents)
$ppoBaselineArgs = @()
foreach ($seed in $seeds) {
    $name = "ppo_${aug}_${preset}_seed${seed}"
    $ppoBaselineArgs += "${name}=${ckptDir}/${name}"
}

# 1. All three algos — strict accuracy across PPO, DG, Soft PPO
Write-Host "=== Strict eval: all baselines (PPO + DG + Soft PPO) ===" -ForegroundColor Magenta
$log = "logs/eval_baselines_v2_strict.log"
if (-not ((Test-Path $log) -and (Select-String -Path $log -Pattern "Evaluation complete" -Quiet))) {
    & $python -u -m rl_captcha.scripts.evaluate_ppo `
        --agent @($allBaselineArgs) `
        --data-dir "data/" --reward-preset $preset `
        --episodes 500 --split test --split-seed 42 `
        --eval-seeds 42 123 456 789 1024 `
        --challenge-as-fp | Tee-Object -FilePath $log
}

# 2. All ablations in strict mode — reveals which ablations challenge humans more
$ablationNames = @(
    "no_hp_bonus", "high_hp_bonus", "strict_fp", "no_continue_cost",
    "small_lstm", "large_lstm", "deep_lstm", "single_view"
)

$total = $ablationNames.Count
$run   = 0
Write-Host ""
Write-Host "=== Strict eval: all ablations ($total types x 5 seeds) ===" -ForegroundColor Magenta

foreach ($ablName in $ablationNames) {
    $run++
    $log = "logs/eval_ablation_${ablName}_strict.log"
    Write-Host "[$run/$total] $ablName" -ForegroundColor Cyan

    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Evaluation complete" -Quiet)) {
        Write-Host "  Skipping - already complete" -ForegroundColor Yellow
        continue
    }

    $ablArgs = @()
    foreach ($seed in $seeds) {
        $name = "ppo_${aug}_${preset}_ablation_${ablName}_seed${seed}"
        $ablArgs += "${name}=${ckptDir}/${name}"
    }

    & $python -u -m rl_captcha.scripts.evaluate_ppo `
        --agent @($ppoBaselineArgs + $ablArgs) `
        --data-dir "data/" --reward-preset $preset `
        --episodes 500 --split test --split-seed 42 `
        --eval-seeds 42 123 456 789 1024 `
        --challenge-as-fp | Tee-Object -FilePath $log

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED" -ForegroundColor Red
    } else {
        Write-Host "  Done -> $log" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Strict evals done." -ForegroundColor Green

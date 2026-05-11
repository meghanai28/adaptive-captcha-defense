# Evaluate ablation models and compare against the ppo_advaug_v2 baseline.
# Each eval log covers all 5 training seeds for one ablation type, plus the
# baseline seeds, so the comparison table is self-contained.
#
# Run from src/ directory:  .\eval_ablations.ps1

$python  = ".\venv\Scripts\python"
$ckptDir = "rl_captcha/agent/checkpoints"
$seeds   = @(42, 123, 456, 789, 1024)
$preset  = "v2"
$aug     = "advaug"

New-Item -ItemType Directory -Force -Path logs | Out-Null

# Ablation definitions — extra eval args needed for architecture ablations
# so the agent is instantiated with the same architecture it was trained with.
$ablations = @(
    @{ name = "no_hp_bonus";      evalArgs = @() },
    @{ name = "high_hp_bonus";    evalArgs = @() },
    @{ name = "strict_fp";        evalArgs = @() },
    @{ name = "no_continue_cost"; evalArgs = @() },
    @{ name = "small_lstm";       evalArgs = @("--lstm-hidden-size", "64") },
    @{ name = "large_lstm";       evalArgs = @("--lstm-hidden-size", "256") },
    @{ name = "deep_lstm";        evalArgs = @("--lstm-num-layers",  "2") },
    @{ name = "single_view";      evalArgs = @("--max-windows", "1", "--random-window-subsample") }
)

# Baseline agent args (ppo_advaug_v2, all 5 seeds)
$baselineArgs = @()
foreach ($seed in $seeds) {
    $name = "ppo_${aug}_${preset}_seed${seed}"
    $baselineArgs += "${name}=${ckptDir}/${name}"
}

function Run-AblationEval($ablName, $agentArgs, $log, $extraEvalArgs) {
    if ((Test-Path $log) -and ((Select-String -Path $log -Pattern "Evaluation complete" -Quiet) -or (Select-String -Path $log -Pattern "Best F1:" -Quiet))) {
        Write-Host "  Skipping $ablName - already done" -ForegroundColor Yellow
        return
    }
    Write-Host "  Evaluating: $ablName ..." -ForegroundColor Cyan
    $baseArgs = @(
        "-u", "-m", "rl_captcha.scripts.evaluate_ppo",
        "--agent"
    ) + $agentArgs + @(
        "--data-dir",    "data/",
        "--reward-preset", $preset,
        "--episodes",    "500",
        "--split",       "test",
        "--split-seed",  "42",
        "--eval-seeds",  "42", "123", "456", "789", "1024"
    ) + $extraEvalArgs
    & $python @baseArgs | Tee-Object -FilePath $log
    Write-Host "  Done -> $log" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== ABLATION EVALUATION ===" -ForegroundColor Magenta
Write-Host ""

foreach ($abl in $ablations) {
    $ablName = $abl.name

    # Build agent args: baseline seeds + ablation seeds
    $ablArgs = @()
    foreach ($seed in $seeds) {
        $name = "ppo_${aug}_${preset}_ablation_${ablName}_seed${seed}"
        $ablArgs += "${name}=${ckptDir}/${name}"
    }
    $allAgentArgs = $baselineArgs + $ablArgs

    $log = "logs/eval_ablation_${ablName}.log"
    Run-AblationEval $ablName $allAgentArgs $log $abl.evalArgs
}

# Also run a combined ablation comparison (all ablations + baseline in one log)
Write-Host ""
Write-Host "=== COMBINED ABLATION SUMMARY ===" -ForegroundColor Magenta
$combinedLog = "logs/eval_ablations_combined.log"
if ((Test-Path $combinedLog) -and ((Select-String -Path $combinedLog -Pattern "Evaluation complete" -Quiet) -or (Select-String -Path $combinedLog -Pattern "Best F1:" -Quiet))) {
    Write-Host "  Skipping combined summary - already done" -ForegroundColor Yellow
} else {
    # One representative seed (seed42) per ablation type for the combined view
    $combinedAgentArgs = $baselineArgs  # start with all baseline seeds
    foreach ($abl in $ablations) {
        $ablName = $abl.name
        # Use seed42 only for the combined overview (reduce eval time)
        $name = "ppo_${aug}_${preset}_ablation_${ablName}_seed42"
        $combinedAgentArgs += "${name}=${ckptDir}/${name}"
    }
    Write-Host "  Running combined ablation overview (seed42 per ablation)..." -ForegroundColor Cyan
    & $python -u -m rl_captcha.scripts.evaluate_ppo `
        --agent @combinedAgentArgs `
        --data-dir "data/" `
        --reward-preset $preset `
        --episodes 500 `
        --split test `
        --split-seed 42 `
        --eval-seeds 42 123 456 789 1024 | Tee-Object -FilePath $combinedLog
    Write-Host "  Done -> $combinedLog" -ForegroundColor Green
}

Write-Host ""
Write-Host "All ablation evaluations complete." -ForegroundColor Green

# Sensitivity analysis: sweep reward/challenge parameters against trained agents.
# Agents evaluated: ppo, dg, soft_ppo -- advaug, v2 preset -- all 5 seeds.
# Run from src/ directory:  .\eval_sensitivity.ps1

$python  = ".\venv\Scripts\python"
$ckptDir = "rl_captcha/agent/checkpoints"
$algos   = @("ppo", "dg", "soft_ppo")
$seeds   = @(42, 123, 456, 789, 1024)
$aug     = "advaug"
$preset  = "v2"

New-Item -ItemType Directory -Force -Path logs | Out-Null

# Build agent args for advaug_v2 (all 3 algos x 5 seeds = 15 agents)
$agentArgs = @()
foreach ($algo in $algos) {
    foreach ($seed in $seeds) {
        $name = "${algo}_${aug}_${preset}_seed${seed}"
        $path = "$ckptDir/$name"
        $agentArgs += "${name}=${path}"
    }
}

$sweeps = @(
    "honeypot_info_bonus",
    "reward_direct_block_bot",
    "penalty_block_human",
    "penalty_bot_missed_allow",
    "easy_puzzle_bot_pass",
    "hard_puzzle_bot_pass",
    "hard_puzzle_human_pass",
    "tier5_honeypot_rate",
    "all_honeypot_rates"
)

Write-Host ""
Write-Host "=== SENSITIVITY ANALYSIS (advaug_v2 agents, 9 sweeps) ===" -ForegroundColor Magenta
Write-Host "  Agents: $($agentArgs.Count) total ($($algos.Count) algos x $($seeds.Count) seeds)"
Write-Host ""

foreach ($sweep in $sweeps) {
    $log = "logs/sensitivity_${sweep}.log"
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Sensitivity complete:" -Quiet)) {
        Write-Host "  Skipping $sweep - already done" -ForegroundColor Yellow
        continue
    }
    Write-Host "  Sweeping: $sweep ..." -ForegroundColor Cyan
    & $python -u -m rl_captcha.scripts.sensitivity_analysis `
        --data-dir "data/" `
        --reward-preset $preset `
        --sweep $sweep `
        --agent @agentArgs `
        --episodes 500 `
        --split test `
        --split-seed 42 `
        --eval-seeds 42 123 456 789 1024 | Tee-Object -FilePath $log
    Write-Host "  Done -> $log" -ForegroundColor Green
    Write-Host ""
}

Write-Host ""
Write-Host "All sensitivity sweeps complete." -ForegroundColor Green

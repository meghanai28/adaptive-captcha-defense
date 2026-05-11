# Ablation study: 8 ablation types x 5 seeds = 40 training runs.
# All ablations use PPO + adversarial augmentation + v2 reward preset as base.
#
# Ablation groups:
#   Reward signal: no_hp_bonus, high_hp_bonus, strict_fp, no_continue_cost
#   Architecture:  small_lstm, large_lstm, deep_lstm, single_view
#
# Run from src/ directory:  .\train_ablations.ps1

$python  = ".\venv\Scripts\python"
$ckptDir = "rl_captcha/agent/checkpoints"
$seeds   = @(42, 123, 456, 789, 1024)
$preset  = "v2"
$aug     = "advaug"

New-Item -ItemType Directory -Force -Path logs | Out-Null

$ablations = @(
    # Reward ablations
    @{ name = "no_hp_bonus";      extraArgs = @() },
    @{ name = "high_hp_bonus";    extraArgs = @() },
    @{ name = "strict_fp";        extraArgs = @() },
    @{ name = "no_continue_cost"; extraArgs = @() },
    # Architecture ablations
    @{ name = "small_lstm";       extraArgs = @() },
    @{ name = "large_lstm";       extraArgs = @() },
    @{ name = "deep_lstm";        extraArgs = @() },
    @{ name = "single_view";      extraArgs = @() }
)

$total = $ablations.Count * $seeds.Count
$run   = 0

Write-Host ""
Write-Host "=== ABLATION TRAINING: $total runs ($($ablations.Count) ablations x $($seeds.Count) seeds) ===" -ForegroundColor Magenta
Write-Host ""

foreach ($abl in $ablations) {
    foreach ($seed in $seeds) {
        $run++
        $name = "ppo_${aug}_${preset}_ablation_$($abl.name)_seed${seed}"
        $ckpt = "$ckptDir/$name"
        $log  = "logs/${name}_training.log"

        Write-Host "[$run/$total] $name" -ForegroundColor Cyan

        if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Training complete\." -Quiet)) {
            Write-Host "  Skipping - already fully trained" -ForegroundColor Yellow
            continue
        }

        $argList = @(
            "-u", "-m", "rl_captcha.scripts.train_ppo",
            "--algorithm",       "ppo",
            "--reward-preset",   $preset,
            "--adversarial-augment",
            "--ablation",        $abl.name,
            "--train-seed",      $seed,
            "--data-dir",        "data/",
            "--save-path",       $ckpt,
            "--total-timesteps", "500000"
        ) + $abl.extraArgs

        & $python @argList | Tee-Object -FilePath $log

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAILED - check $log" -ForegroundColor Red
        } else {
            Write-Host "  Done -> $ckpt" -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "All $total ablation runs complete." -ForegroundColor Green

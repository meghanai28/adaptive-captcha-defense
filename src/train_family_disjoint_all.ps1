# Family-disjoint generalization: train agents with each individual bot
# family held out, then evaluate only on sessions from that family.
# Mirrors src/classifier/scripts/evaluate_family_disjoint.py so the RL
# vs XGBoost comparison is symmetric at family granularity (10 families).
#
# Run from src/ directory:  .\train_family_disjoint_all.ps1
# Expected wall time: ~12-15 hrs (50 seeds x 500k steps each)

$python  = ".\venv\Scripts\python"
$ckptDir = "rl_captcha/agent/checkpoints"
$seeds   = @(42, 123, 456, 789, 1024)
$preset  = "v2"
$aug     = "advaug"

New-Item -ItemType Directory -Force -Path logs | Out-Null

# 10 families matching classifier FAMILY_DISPLAY in evaluate_family_disjoint.py
$families = @(
    "linear",
    "tabber",
    "speedrun",
    "scripted",
    "stealth",
    "slow",
    "erratic",
    "semi_auto",
    "trace_conditioned",
    "llm"
)

$totalConfigs = $families.Count
$configRun    = 0

foreach ($family in $families) {
    $configRun++
    $tag = "disjoint_family_${family}"

    Write-Host ""
    Write-Host "=== [$configRun/$totalConfigs] $tag  (family '$family' held out) ===" -ForegroundColor Magenta

    $total = $seeds.Count
    $run   = 0

    foreach ($seed in $seeds) {
        $run++
        $name = "ppo_${aug}_${preset}_${tag}_seed${seed}"
        $ckpt = "$ckptDir/$name"
        $log  = "logs/${name}_training.log"

        Write-Host "  [$run/$total] $name" -ForegroundColor Cyan

        if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Training complete\." -Quiet)) {
            Write-Host "    Skipping - already fully trained" -ForegroundColor Yellow
            continue
        }

        $trainArgs = @(
            "-u", "-m", "rl_captcha.scripts.train_ppo",
            "--algorithm", "ppo",
            "--reward-preset", $preset,
            "--adversarial-augment",
            "--held-out-families", $family,
            "--train-seed", $seed,
            "--data-dir", "data/",
            "--save-path", $ckpt,
            "--total-timesteps", "500000"
        )

        & $python @trainArgs | Tee-Object -FilePath $log

        if ($LASTEXITCODE -ne 0) {
            Write-Host "    FAILED - check $log" -ForegroundColor Red
        } else {
            Write-Host "    Done -> $ckpt" -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "All $totalConfigs family-disjoint training configs complete." -ForegroundColor Green
Write-Host "Next: run .\eval_family_disjoint_all.ps1" -ForegroundColor Cyan

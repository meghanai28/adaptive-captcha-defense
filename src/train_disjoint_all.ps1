# Disjoint generalization: train agents with each tier (or tier combo) held out.
# Complements train_disjoint.ps1 (Tier 5 only) by covering T1-T4 and combos.
# Each model is tested by eval_disjoint_all.ps1 on its held-out tier sessions only.
#
# Run from src/ directory:  .\train_disjoint_all.ps1
# Expected wall time: ~6-8 hrs (30 seeds x 500k steps each)

$python  = ".\venv\Scripts\python"
$ckptDir = "rl_captcha/agent/checkpoints"
$seeds   = @(42, 123, 456, 789, 1024)
$preset  = "v2"
$aug     = "advaug"

New-Item -ItemType Directory -Force -Path logs | Out-Null

# Each entry: @{ tag = "disjoint_tierX"; tiers = @(X,...) }
$configs = @(
    @{ tag = "disjoint_tier1";   tiers = @(1)       }
    @{ tag = "disjoint_tier2";   tiers = @(2)       }
    @{ tag = "disjoint_tier3";   tiers = @(3)       }
    @{ tag = "disjoint_tier4";   tiers = @(4)       }
    @{ tag = "disjoint_tier45";  tiers = @(4, 5)    }
    @{ tag = "disjoint_tier345"; tiers = @(3, 4, 5) }
)

$totalConfigs = $configs.Count
$configRun    = 0

foreach ($c in $configs) {
    $configRun++
    $label = $c.tag -replace "disjoint_", ""
    Write-Host ""
    Write-Host "=== [$configRun/$totalConfigs] $($c.tag)  (tiers $($c.tiers -join '+') held out) ===" -ForegroundColor Magenta

    $total = $seeds.Count
    $run   = 0

    foreach ($seed in $seeds) {
        $run++
        $name = "ppo_${aug}_${preset}_$($c.tag)_seed${seed}"
        $ckpt = "$ckptDir/$name"
        $log  = "logs/${name}_training.log"

        Write-Host "  [$run/$total] $name" -ForegroundColor Cyan

        if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Training complete\." -Quiet)) {
            Write-Host "    Skipping - already fully trained" -ForegroundColor Yellow
            continue
        }

        # Build arg list so multi-tier values expand as separate CLI tokens
        $trainArgs = @(
            "-u", "-m", "rl_captcha.scripts.train_ppo",
            "--algorithm", "ppo",
            "--reward-preset", $preset,
            "--adversarial-augment",
            "--held-out-tiers"
        ) + $c.tiers + @(
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
Write-Host "All $totalConfigs disjoint training configs complete." -ForegroundColor Green
Write-Host "Next: run .\eval_disjoint_all.ps1" -ForegroundColor Cyan

# Evaluate all disjoint-trained agents on their respective held-out tier sessions.
# Must be run AFTER train_disjoint_all.ps1 (and train_disjoint.ps1 for Tier 5).
#
# Each log is self-contained: baseline agents + disjoint agents evaluated on
# the tiers that were held out during disjoint training.
#
# Run from src/ directory:  .\eval_disjoint_all.ps1

$python  = ".\venv\Scripts\python"
$ckptDir = "rl_captcha/agent/checkpoints"
$seeds   = @(42, 123, 456, 789, 1024)
$preset  = "v2"
$aug     = "advaug"

New-Item -ItemType Directory -Force -Path logs | Out-Null

# Baseline agents (trained WITH all tiers)
$baselineArgs = @()
foreach ($seed in $seeds) {
    $name = "ppo_${aug}_${preset}_seed${seed}"
    $baselineArgs += "${name}=${ckptDir}/${name}"
}

# Each entry: @{ tag = "disjoint_tierX"; tiers = @(X,...) }
$configs = @(
    @{ tag = "disjoint_tier1";   tiers = @(1)       }
    @{ tag = "disjoint_tier2";   tiers = @(2)       }
    @{ tag = "disjoint_tier3";   tiers = @(3)       }
    @{ tag = "disjoint_tier4";   tiers = @(4)       }
    @{ tag = "disjoint_tier5";   tiers = @(5)       }
    @{ tag = "disjoint_tier45";  tiers = @(4, 5)    }
    @{ tag = "disjoint_tier345"; tiers = @(3, 4, 5) }
)

$totalConfigs = $configs.Count
$configRun    = 0

foreach ($c in $configs) {
    $configRun++
    $logSuffix = $c.tag -replace "disjoint_", ""
    $log = "logs/eval_disjoint_${logSuffix}.log"

    Write-Host ""
    Write-Host "=== [$configRun/$totalConfigs] $($c.tag)  (eval on tiers $($c.tiers -join '+') only) ===" -ForegroundColor Magenta

    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Evaluation complete" -Quiet)) {
        Write-Host "  Already done -> $log" -ForegroundColor Yellow
        continue
    }

    # Disjoint agents for this tier config
    $disjointArgs = @()
    foreach ($seed in $seeds) {
        $name = "ppo_${aug}_${preset}_$($c.tag)_seed${seed}"
        $disjointArgs += "${name}=${ckptDir}/${name}"
    }

    # Check that at least one checkpoint exists before trying to eval
    $firstCkpt = "$ckptDir/ppo_${aug}_${preset}_$($c.tag)_seed42"
    if (-not (Test-Path $firstCkpt)) {
        Write-Host "  SKIP - no checkpoint found at $firstCkpt (run train_disjoint_all.ps1 first)" -ForegroundColor Red
        continue
    }

    # Build eval arg list (held-out-tiers expands as separate tokens)
    $evalArgs = @(
        "-u", "-m", "rl_captcha.scripts.evaluate_ppo",
        "--agent"
    ) + @($baselineArgs + $disjointArgs) + @(
        "--data-dir", "data/",
        "--reward-preset", $preset,
        "--episodes", "500",
        "--split", "test",
        "--split-seed", "42",
        "--held-out-tiers"
    ) + $c.tiers + @(
        "--eval-seeds", "42", "123", "456", "789", "1024"
    )

    & $python @evalArgs | Tee-Object -FilePath $log

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED - check $log" -ForegroundColor Red
    } else {
        Write-Host "  Done -> $log" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "All disjoint evals done. Run generate_paper_figures.py to regenerate fig 15." -ForegroundColor Green

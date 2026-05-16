# Evaluate all family-disjoint agents on their respective held-out family
# sessions. Must be run AFTER train_family_disjoint_all.ps1.
#
# Each log compares baseline agents (trained WITH all families) against the
# family-disjoint agents on the sessions of the held-out family only.
# Result table is directly comparable to src/classifier/family_disjoint/.
#
# Run from src/ directory:  .\eval_family_disjoint_all.ps1

$python  = ".\venv\Scripts\python"
$ckptDir = "rl_captcha/agent/checkpoints"
$seeds   = @(42, 123, 456, 789, 1024)
$preset  = "v2"
$aug     = "advaug"

New-Item -ItemType Directory -Force -Path logs | Out-Null

# Baseline agents (trained WITH all families)
$baselineArgs = @()
foreach ($seed in $seeds) {
    $name = "ppo_${aug}_${preset}_seed${seed}"
    $baselineArgs += "${name}=${ckptDir}/${name}"
}

# 10 families matching the classifier evaluation
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
    $log = "logs/eval_disjoint_family_${family}.log"

    Write-Host ""
    Write-Host "=== [$configRun/$totalConfigs] $tag  (eval on family '$family' only) ===" -ForegroundColor Magenta

    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Evaluation complete" -Quiet)) {
        Write-Host "  Already done -> $log" -ForegroundColor Yellow
        continue
    }

    # Disjoint agents for this family
    $disjointArgs = @()
    foreach ($seed in $seeds) {
        $name = "ppo_${aug}_${preset}_${tag}_seed${seed}"
        $disjointArgs += "${name}=${ckptDir}/${name}"
    }

    # Skip if no checkpoints yet
    $firstCkpt = "$ckptDir/ppo_${aug}_${preset}_${tag}_seed42"
    if (-not (Test-Path $firstCkpt)) {
        Write-Host "  SKIP - no checkpoint at $firstCkpt (run train_family_disjoint_all.ps1 first)" -ForegroundColor Red
        continue
    }

    $evalArgs = @(
        "-u", "-m", "rl_captcha.scripts.evaluate_ppo",
        "--agent"
    ) + @($baselineArgs + $disjointArgs) + @(
        "--data-dir", "data/",
        "--reward-preset", $preset,
        "--episodes", "500",
        "--split", "test",
        "--split-seed", "42",
        "--held-out-families", $family,
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
Write-Host "All family-disjoint evals done." -ForegroundColor Green
Write-Host "Result tables now match classifier family-disjoint granularity (10 families)." -ForegroundColor Cyan

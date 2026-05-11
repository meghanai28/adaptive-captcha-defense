# Evaluate human-disjoint PPO agents.
# For each person (A, B): evaluate the held-out model AND the baseline PPO
# on ONLY that person's sessions (the ones never seen during training).
#
# Run from src/ directory:  .\eval_human_disjoint.ps1

$python      = ".\venv\Scripts\python"
$ckptDir     = "rl_captcha/agent/checkpoints"
$seeds       = @(42, 123, 456, 789, 1024)
$preset      = "v2"
$aug         = "advaug"
$personADir  = "C:\Users\megha\Downloads\person A sessions"
$personBDir  = "C:\Users\megha\Downloads\person B sessions"

New-Item -ItemType Directory -Force -Path logs | Out-Null

# Baseline PPO agents (trained on all people)
$baselineArgs = @()
foreach ($seed in $seeds) {
    $name = "ppo_${aug}_${preset}_seed${seed}"
    $baselineArgs += "${name}=${ckptDir}/${name}"
}

$persons = @(
    @{ label = "personA"; dir = $personADir; held = "A" }
    @{ label = "personB"; dir = $personBDir; held = "B" }
)

foreach ($person in $persons) {
    $log = "logs/eval_human_disjoint_$($person.label).log"
    Write-Host ""
    Write-Host "=== Eval held-out $($person.label) ===" -ForegroundColor Magenta

    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Evaluation complete" -Quiet)) {
        Write-Host "  Already done -> $log" -ForegroundColor Yellow
        continue
    }

    # Disjoint agents for this person
    $disjointArgs = @()
    foreach ($seed in $seeds) {
        $name = "ppo_${aug}_${preset}_heldout_$($person.label)_seed${seed}"
        $ckpt = "$ckptDir/$name"
        if (Test-Path $ckpt) {
            $disjointArgs += "${name}=${ckpt}"
        } else {
            Write-Host "  WARNING: checkpoint not found: $ckpt" -ForegroundColor Yellow
        }
    }

    if ($disjointArgs.Count -eq 0) {
        Write-Host "  SKIP - no disjoint checkpoints found (run train_human_disjoint.ps1 first)" -ForegroundColor Red
        continue
    }

    & $python -u -m rl_captcha.scripts.evaluate_ppo `
        --agent @($baselineArgs + $disjointArgs) `
        --data-dir "data/" `
        --reward-preset $preset `
        --episodes 500 `
        --split test `
        --split-seed 42 `
        --eval-seeds 42 123 456 789 1024 `
        --person-a-dir $personADir `
        --person-b-dir $personBDir `
        --held-out-person $person.held | Tee-Object -FilePath $log

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED - check $log" -ForegroundColor Red
    } else {
        Write-Host "  Done -> $log" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Human disjoint eval complete." -ForegroundColor Green

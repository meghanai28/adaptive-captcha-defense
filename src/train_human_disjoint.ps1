# Human disjoint generalization: train PPO with one person's sessions completely
# excluded from training. Run once for Person A held out, once for Person B.
#
# Person A/B sessions are identified by filename — the session files in those
# directories are the same files already in data/human/. Nothing is renamed or moved.
#
# Run from src/ directory:  .\train_human_disjoint.ps1

$python      = ".\venv\Scripts\python"
$ckptDir     = "rl_captcha/agent/checkpoints"
$seeds       = @(42, 123, 456, 789, 1024)
$preset      = "v2"
$aug         = "advaug"
$personADir  = "C:\Users\megha\Downloads\person A sessions"
$personBDir  = "C:\Users\megha\Downloads\person B sessions"

New-Item -ItemType Directory -Force -Path logs | Out-Null

$persons = @(
    @{ label = "personA"; dir = $personADir }
    @{ label = "personB"; dir = $personBDir }
)

foreach ($person in $persons) {
    $heldOut = if ($person.label -eq "personA") { "A" } else { "B" }
    Write-Host ""
    Write-Host "=== Held-out $($person.label) ===" -ForegroundColor Magenta

    $total = $seeds.Count
    $run   = 0
    foreach ($seed in $seeds) {
        $run++
        $name = "ppo_${aug}_${preset}_heldout_$($person.label)_seed${seed}"
        $ckpt = "$ckptDir/$name"
        $log  = "logs/${name}_training.log"

        Write-Host "  [$run/$total] $name" -ForegroundColor Cyan

        if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Training complete\." -Quiet)) {
            Write-Host "    Skipping - already fully trained" -ForegroundColor Yellow
            continue
        }

        & $python -u -m rl_captcha.scripts.train_ppo `
            --algorithm ppo `
            --reward-preset $preset `
            --adversarial-augment `
            --person-a-dir $personADir `
            --person-b-dir $personBDir `
            --held-out-person $heldOut `
            --train-seed $seed `
            --data-dir "data/" `
            --save-path $ckpt `
            --total-timesteps 500000 | Tee-Object -FilePath $log

        if ($LASTEXITCODE -ne 0) {
            Write-Host "    FAILED - check $log" -ForegroundColor Red
        } else {
            Write-Host "    Done -> $ckpt" -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "Training complete. Run .\eval_human_disjoint.ps1 next." -ForegroundColor Green

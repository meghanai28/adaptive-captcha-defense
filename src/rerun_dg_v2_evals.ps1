# Re-run all DG advaug v2 evals with the retrained seed1024 checkpoint.
# Deletes stale logs first so nothing old remains.
# Run from src/ directory: .\rerun_dg_v2_evals.ps1

$python  = ".\venv\Scripts\python"
$ckptDir = "rl_captcha/agent/checkpoints"
$seeds   = @(42, 123, 456, 789, 1024)
$preset  = "v2"
$aug     = "advaug"
$algo    = "dg"

New-Item -ItemType Directory -Force -Path logs | Out-Null

$agentArgs = @()
foreach ($seed in $seeds) {
    $name = "${algo}_${aug}_${preset}_seed${seed}"
    $agentArgs += "${name}=${ckptDir}/${name}"
}

$baseArgs = @(
    "-u", "-m", "rl_captcha.scripts.evaluate_ppo",
    "--agent"
) + $agentArgs + @(
    "--data-dir", "data/",
    "--reward-preset", $preset,
    "--episodes", "500",
    "--split", "test",
    "--split-seed", "42",
    "--eval-seeds", "42", "123", "456", "789", "1024"
)

function Run-Fresh($label, $log, $extraArgs) {
    Write-Host ""
    Write-Host "=== $label ===" -ForegroundColor Magenta
    if (Test-Path $log) { Remove-Item $log -Force }
    & $python @baseArgs @extraArgs | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED -> $log" -ForegroundColor Red
    } else {
        Write-Host "  Done -> $log" -ForegroundColor Green
    }
}

Run-Fresh "Native" "logs/eval_dg_advaug_v2_native.log" @()

Run-Fresh "Cross (v2 in v1 env)" "logs/eval_dg_advaug_v2_in_v1_env.log" @("--reward-preset", "v1")

$tierTests = @(
    @{ label = "tier3";   tiers = @("3")           }
    @{ label = "tier4";   tiers = @("4")           }
    @{ label = "tier5";   tiers = @("5")           }
    @{ label = "tier45";  tiers = @("4", "5")      }
    @{ label = "tier345"; tiers = @("3", "4", "5") }
)
foreach ($t in $tierTests) {
    $log = "logs/eval_dg_advaug_v2_heldout_$($t.label).log"
    Run-Fresh "Held-out $($t.label)" $log (@("--reward-preset", $preset, "--held-out-tiers") + $t.tiers)
}

foreach ($family in @("stealth", "replay", "llm", "semi_auto", "trace_conditioned")) {
    $log = "logs/eval_dg_advaug_v2_heldout_${family}.log"
    Run-Fresh "Held-out $family" $log @("--reward-preset", $preset, "--held-out-families", $family)
}

Run-Fresh "Augtest" "logs/eval_dg_advaug_v2_augtest.log" @("--reward-preset", $preset, "--include-augmented")

Write-Host ""
Write-Host "All DG v2 evals done." -ForegroundColor Green

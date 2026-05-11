# Full evaluation suite — covers all reviewer requests:
#   1. Native eval: each model evaluated in its own reward environment
#   2. Cross eval: each model evaluated in the OTHER reward environment
#   3. Held-out tier 3, 4, 5 generalization tests (native env)
#   4. Held-out bot-family generalization tests (native env)
#   5. Augmented test set eval (native env)
#
# Run from src/ directory:  .\eval_all.ps1

$python  = ".\venv\Scripts\python"
$ckptDir = "rl_captcha/agent/checkpoints"
$algos   = @("ppo", "dg", "soft_ppo")
$presets = @("v1", "v2")
$augs    = @("noaug", "advaug")
$seeds   = @(42, 123, 456, 789, 1024)

New-Item -ItemType Directory -Force -Path logs | Out-Null

function Get-AgentArgs($algo, $aug, $preset) {
    $args = @()
    foreach ($seed in $seeds) {
        $name = "${algo}_${aug}_${preset}_seed${seed}"
        $path = "$ckptDir/$name"
        $args += "${name}=${path}"
    }
    return $args
}

function Run-Eval($label, $log, $agentArgs, $extraArgs) {
    if ((Test-Path $log) -and ((Select-String -Path $log -Pattern "Evaluation complete" -Quiet) -or (Select-String -Path $log -Pattern "Best F1:" -Quiet))) {
        Write-Host "  Skipping $label - already done" -ForegroundColor Yellow
        return
    }
    Write-Host "  $label..." -ForegroundColor Cyan
    $baseArgs = @("-u", "-m", "rl_captcha.scripts.evaluate_ppo",
                  "--agent") + $agentArgs +
                @("--data-dir", "data/",
                  "--episodes", "500",
                  "--split", "test",
                  "--split-seed", "42",
                  "--eval-seeds", "42", "123", "456", "789", "1024")
    & $python @baseArgs @extraArgs | Tee-Object -FilePath $log
    Write-Host "  Done -> $log" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 1. NATIVE EVAL — each model in its own reward environment
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== PHASE 1: Native eval (model in its own reward environment) ===" -ForegroundColor Magenta
foreach ($preset in $presets) {
    foreach ($aug in $augs) {
        foreach ($algo in $algos) {
            $label     = "${algo}_${aug}_${preset} in ${preset} env"
            $log       = "logs/eval_${algo}_${aug}_${preset}_native.log"
            $agentArgs = Get-AgentArgs $algo $aug $preset
            Run-Eval $label $log $agentArgs @("--reward-preset", $preset)
        }
    }
}

# ---------------------------------------------------------------------------
# 2. CROSS EVAL — each model in the OTHER reward environment
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== PHASE 2: Cross eval (model in opposite reward environment) ===" -ForegroundColor Magenta
foreach ($preset in $presets) {
    $otherPreset = if ($preset -eq "v1") { "v2" } else { "v1" }
    foreach ($aug in $augs) {
        foreach ($algo in $algos) {
            $label     = "${algo}_${aug}_${preset} in ${otherPreset} env"
            $log       = "logs/eval_${algo}_${aug}_${preset}_in_${otherPreset}_env.log"
            $agentArgs = Get-AgentArgs $algo $aug $preset
            Run-Eval $label $log $agentArgs @("--reward-preset", $otherPreset)
        }
    }
}

# ---------------------------------------------------------------------------
# 3. HELD-OUT TIER GENERALIZATION — advaug models, native env
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== PHASE 3: Held-out tier generalization ===" -ForegroundColor Magenta
$tierTests = @(
    @{ tiers = @("3");       label = "tier3"   },
    @{ tiers = @("4");       label = "tier4"   },
    @{ tiers = @("5");       label = "tier5"   },
    @{ tiers = @("4", "5");  label = "tier45"  },
    @{ tiers = @("3","4","5"); label = "tier345" }
)
foreach ($preset in $presets) {
    foreach ($algo in $algos) {
        foreach ($test in $tierTests) {
            $tlabel    = $test.label
            $log       = "logs/eval_${algo}_advaug_${preset}_heldout_${tlabel}.log"
            $agentArgs = Get-AgentArgs $algo "advaug" $preset
            $tierArgs  = @("--held-out-tiers") + $test.tiers
            Run-Eval "${algo}_advaug_${preset} held-out $tlabel" $log $agentArgs (@("--reward-preset", $preset) + $tierArgs)
        }
    }
}

# ---------------------------------------------------------------------------
# 4. HELD-OUT FAMILY GENERALIZATION — advaug models, native env
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== PHASE 4: Held-out family generalization ===" -ForegroundColor Magenta
$families = @("stealth", "replay", "llm", "semi_auto", "trace_conditioned")
foreach ($preset in $presets) {
    foreach ($algo in $algos) {
        foreach ($family in $families) {
            $log       = "logs/eval_${algo}_advaug_${preset}_heldout_${family}.log"
            $agentArgs = Get-AgentArgs $algo "advaug" $preset
            Run-Eval "${algo}_advaug_${preset} held-out $family" $log $agentArgs @("--reward-preset", $preset, "--held-out-families", $family)
        }
    }
}

# ---------------------------------------------------------------------------
# 5. AUGMENTED TEST SET — all combos, native env
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== PHASE 5: Augmented test set eval ===" -ForegroundColor Magenta
foreach ($preset in $presets) {
    foreach ($aug in $augs) {
        foreach ($algo in $algos) {
            $log       = "logs/eval_${algo}_${aug}_${preset}_augtest.log"
            $agentArgs = Get-AgentArgs $algo $aug $preset
            Run-Eval "${algo}_${aug}_${preset} augmented test" $log $agentArgs @("--reward-preset", $preset, "--include-augmented")
        }
    }
}

Write-Host ""
Write-Host "All evaluations complete." -ForegroundColor Green

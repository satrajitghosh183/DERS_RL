#!/usr/bin/env bash
# Overnight guardian for the hands-off pipeline. The known crash (prune race) is
# fixed; this is backup. Restarts run_pipeline.sh (resume=1, near-instant no-op for
# already-done stages) if Stage-1 training deadlocks (step frozen >6min) or the
# orchestrator dies before "PIPELINE DONE". Conservative thresholds; logs every act.
LOG=~/pipeline/watchdog.log
say(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
relaunch(){ pkill -9 -f olmo_train; pkill -9 -f launch_multigpu; pkill -9 -f run_pipeline.sh; sleep 6
            setsid bash ~/run_pipeline.sh > ~/pipeline/run.log 2>&1 </dev/null & disown
            say "RELAUNCHED run_pipeline (resume=1)"; sleep 90; }
say "=== watchdog start (pid $$) ==="
last=""; stall=0
while true; do
  sleep 120
  grep -q "PIPELINE DONE" ~/pipeline/run.log 2>/dev/null && { say "PIPELINE DONE — watchdog exit"; exit 0; }
  orch=$(pgrep -f run_pipeline.sh | grep -v "^$$\$")
  if ! grep -q "STAGE 2" ~/pipeline/run.log 2>/dev/null; then
    cur=$(grep -aoE "Step [0-9]+/20000" ~/pipeline/01_slm_train.log 2>/dev/null | tail -1)
    ranks=$(pgrep -cf olmo_train)
    if [ -n "$cur" ] && [ "$cur" = "$last" ]; then stall=$((stall+1)); else stall=0; fi
    last="$cur"
    if [ "$stall" -ge 3 ]; then
      say "STAGE1 HANG (step frozen 6min at '$cur', ranks=$ranks) — restarting"; relaunch; stall=0; last=""
    fi
  else
    if [ -z "$orch" ]; then say "orchestrator gone past Stage1 w/o DONE — restarting"; relaunch; fi
  fi
done

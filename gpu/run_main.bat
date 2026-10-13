@echo off
REM Main experiments (no 14B), in priority order. Run from Anaconda Prompt (env "cpd") inside Documents\cpd_gpu:
REM     run_main.bat
REM Every step saves its own results3_*.json; re-running skips finished steps.
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
echo ===== 1/6 GPU timing on this RTX 5000 Ada (~10 min) =====
python cpd_ada.py --tag hw --stages timing
echo ===== 2/6 Qwen2.5-7B: perplexity, LAMBADA, exactness, Byzantine, analog baseline, coded attention (~1 h) =====
python cpd_ada.py --tag q7 --stages quality attention --models Qwen/Qwen2.5-7B --analog
echo ===== 3/6 Qwen2.5-7B: seven zero-shot benchmarks + coded check (~1.25 h) =====
python cpd_ada.py --tag q7 --stages tasks --models Qwen/Qwen2.5-7B --task-n 500 --w8a16-task-n 200 --coded-task-n 50
echo ===== 4/6 Qwen2.5-7B: PIN-recovery privacy attack (~1.25 h) =====
python cpd_ada.py --tag q7 --stages privacy --models Qwen/Qwen2.5-7B
echo ===== 5/6 OLMo-2-7B (second model family): quality + analog + benchmarks (~1.5 h) =====
python cpd_ada.py --tag o7 --stages quality tasks --models allenai/OLMo-2-1124-7B --analog --task-n 300 --w8a16-task-n 100 --coded-task-n 30
echo ===== 6/6 Qwen2.5-7B: GSM8K with coded generation (~1 h) =====
python cpd_ada.py --tag q7 --stages generate --models Qwen/Qwen2.5-7B --gen-n 100 --gen-w8a16-n 40 --gen-coded-n 10 --gen-max 200
echo ===== ALL STEPS DONE =====

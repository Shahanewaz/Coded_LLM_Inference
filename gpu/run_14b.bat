@echo off
REM Qwen2.5-14B experiments (run later). Close Chrome/Teams/Zoom first: the model needs almost all 32 GB.
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
echo ===== 1/2 Qwen2.5-14B: perplexity, LAMBADA, exactness, Byzantine (~45 min) =====
python cpd_ada.py --tag q14 --stages quality --models Qwen/Qwen2.5-14B --windows 16
echo ===== 2/2 Qwen2.5-14B: seven zero-shot benchmarks + coded check (~1.25 h) =====
python cpd_ada.py --tag q14 --stages tasks --models Qwen/Qwen2.5-14B --task-n 300 --w8a16-task-n 100 --coded-task-n 20
echo ===== ALL STEPS DONE =====

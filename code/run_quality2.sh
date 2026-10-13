#!/bin/bash
cd /home/claude/code
while pgrep -f "windows 64 --lambada 500" >/dev/null; do sleep 20; done
python3 -I evalppl.py w8a8 --windows 64 --lambada 500
python3 -I evalppl.py ff:8,4,2,8 --windows 4
python3 -I evalppl.py bf16 --windows 64 --lambada 500
python3 -I evalppl.py fp16 --windows 64
for f in bf16 fp16 fp32; do for b in 1 10 100 1000 10000; do python3 -I evalppl.py analog:$f,$b --windows 16; done; done
python3 -I evalppl.py fp32 --windows 16
python3 -I evalppl.py w8a8 --windows 16
python3 -I evalppl.py w8a12 --windows 16
python3 -I evalppl.py w8a16 --windows 16
python3 -I evalppl.py bf16 --windows 16
echo DONE

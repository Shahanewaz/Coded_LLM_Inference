#!/bin/bash
cd /home/claude/code
export OPENBLAS_NUM_THREADS=1
for f in bf16 fp32 fp16; do for b in 1 10 100 1000 10000; do python3 -I evalppl.py analog:$f,$b --windows 16; done; done
for c in fp32 w8a8 w8a12 w8a16 bf16 fp16; do python3 -I evalppl.py $c --windows 16; done
echo DONE

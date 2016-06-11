import os
import sys

args = sys.argv
if len(args) > 1:
    queue = args[1]
else:
    queue = "low.q"
walks = 60
if len(args) > 2:
    walks = int(args[2])
    if (walks % 3 != 0):
        raise "Walks must be divisible by 3"
    
directory = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
template = '''#!/bin/bash
#$ -S /bin/bash
#$ -pe threaded 1
#$ -M samuelreay@gmail.com
#$ -N %s
#$ -m abe
#$ -q %s
#$ -V
#$ -l gpu=0
#$ -t 1:%d
#$ -tc 61
#$ -wd /home/uqshint1/thesis/%s/out_files
#$ -o /home/uqshint1/thesis/%s/out_files/$JOB_NAME.$JOB_ID.out
#$ -e /home/uqshint1/thesis/%s/out_files/errors
IDIR=/home/uqshint1/thesis/%s

export OMP_NUM_THREADS="1" # set this for OpenMP threads control
export MKL_NUM_THREADS="1" # set this for Intel MKL threads control
echo 'running with OMP_NUM_THREADS =' $OMP_NUM_THREADS
echo 'running with MKL_NUM_THREADS =' $MKL_NUM_THREADS
echo 'running with NSLOTS=' $NSLOTS # number of SGE calcs

PROG=cambMCMC.py
PARAMS=`expr $SGE_TASK_ID - 1`
cd $IDIR
python $PROG $PARAMS'''

n = "jobscript_%s" % (queue)
t = template % (directory, queue, walks+1, directory, directory, directory, directory)
with open(n, 'w') as f:
    f.write(t)
print(n)
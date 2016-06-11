#!/bin/bash
n=`python doJob.py "$@"`
qsub $n


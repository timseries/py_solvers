#!/bin/sh
# Simple Grid Engine job submission script
qsub -l mem_free=18G,mem_grab=18G -m e job.sh

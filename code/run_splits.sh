#!/bin/bash

python generate_sample_splits.py 1733 5 4 100 ../../../data/features/SMRI/ADNI/labels_allgroups.txt ../in/SampleSplits/ADNI/allgroups/
python generate_sample_splits.py 1733 5 4 100 ../../../data/features/SMRI/ADNI/labels_3way.txt ../in/SampleSplits/ADNI/3way/
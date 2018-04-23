#!/bin/bash


rgb_score=$1
flow_score=
python ./merge.py --rgb ${rgb_score}

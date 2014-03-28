#! /usr/bin/env bash

URL="https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits-orig.tra.Z"
TMP="/tmp/pendigits.z"
curl $URL > $TMP

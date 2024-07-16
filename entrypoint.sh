#!/bin/bash

if [ "$1" = "train" ]; then
    python ModelDevelopment.py train
elif [ "$1" = "predict" ]; then
    python NewDataPrediction.py
else
    echo "Invalid argument. Use 'train' or 'predict'."
fi
#!/bin/bash

arg1=$1
jupyter nbconvert --execute --to notebook --inplace $arg1
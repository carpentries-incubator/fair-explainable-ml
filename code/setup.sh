#!/bin/bash
# Run this script to move the MEPS data to the correct location for the
# aif360 library. You may need to adjust the filepaths in lines 12-13 if 
# you downloaded the data to a different directory.

pip install aif360

# find place where libraries are stored
PYTHONLOC=$(pip show numpy | awk '/Location:/{print $2}')
echo $PYTHONLOC

cp ../data/meps/h181.csv $PYTHONLOC/aif360/data/raw/meps/h181.csv
cp ../data/meps/h192.csv $PYTHONLOC/aif360/data/raw/meps/h192.csv


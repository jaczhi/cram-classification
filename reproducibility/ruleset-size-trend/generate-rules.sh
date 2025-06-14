#!/bin/bash

mkdir -p ../acl-list

for i in $(seq -w 5 5 100)
do
   # Calculate the value by multiplying i by 1000
   value=$((10#$i * 1000))
   
   # Run the command with the calculated value
   ./db_generator -bc ../parameter_files/acl1_seed $value 2 0.5 -0.1 "../acl-list/acl-list-$i"
done

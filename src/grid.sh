#!/bin/sh

for m in 30 28 26 24 22 20
do
	for horizon in 8 9
	do
		echo "$horizon"
		echo "$m"
		out="outputs_nav/$horizon/nav_new_$m.txt"
		python rspmn_new1.py --dataset navigation_new --mi_threshold $m --horizon $horizon --samples 500000 --num_vars 9 &>> $out
	done
done

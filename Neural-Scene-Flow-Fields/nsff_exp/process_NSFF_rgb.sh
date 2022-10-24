#!/bin/bash

python run_nerf.py --config configs/config_ska*_0.txt --render_2D

python run_nerf.py --config configs/config_pla*_0.txt --render_2D

python run_nerf.py --config configs/config_*alloon1*_0.txt --render_2D

python run_nerf.py --config configs/config_*alloon2*_0.txt --render_2D

python run_nerf.py --config configs/config_*ynam*_0.txt --render_2D

python run_nerf.py --config configs/config_truc*_0.txt --render_2D

python run_nerf.py --config configs/config_*mbrell*_0.txt --render_2D

python run_nerf.py --config configs/config_*umping*_0.txt --render_2D

#!/bin/bash

export METHOD_NAME=nosal

python run_nerf.py --config configs/config_ska*_$METHOD_NAME*.txt --render_2D

python run_nerf.py --config configs/config_pla*_$METHOD_NAME*.txt --render_2D

python run_nerf.py --config configs/config_*alloon1*_$METHOD_NAME*.txt --render_2D

python run_nerf.py --config configs/config_*alloon2*_$METHOD_NAME*.txt --render_2D

python run_nerf.py --config configs/config_*ynam*_$METHOD_NAME*.txt --render_2D

python run_nerf.py --config configs/config_truc*_$METHOD_NAME*.txt --render_2D

python run_nerf.py --config configs/config_*mbrell*_$METHOD_NAME*.txt --render_2D

python run_nerf.py --config configs/config_*umping*_$METHOD_NAME*.txt --render_2D

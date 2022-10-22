#!/bin/bash

python run_nerf.py --config configs/config_ska*_4.txt --cluster_2D

python run_nerf.py --config configs/config_ska*_4.txt --cluster_2D --render_mode

python run_nerf.py --config configs/config_pla*_4.txt --cluster_2D

python run_nerf.py --config configs/config_pla*_4.txt --cluster_2D --render_mode

python run_nerf.py --config configs/config_*alloon1*_4.txt --cluster_2D

python run_nerf.py --config configs/config_*alloon1*_4.txt --cluster_2D --render_mode

python run_nerf.py --config configs/config_*alloon2*_4.txt --cluster_2D

python run_nerf.py --config configs/config_*alloon2*_4.txt --cluster_2D --render_mode

python run_nerf.py --config configs/config_*ynam*_4.txt --cluster_2D

python run_nerf.py --config configs/config_*ynam*_4.txt --cluster_2D --render_mode

python run_nerf.py --config configs/config_truc*_4.txt --cluster_2D

python run_nerf.py --config configs/config_truc*_4.txt --cluster_2D --render_mode

python run_nerf.py --config configs/config_*mbrell*_4.txt --cluster_2D

python run_nerf.py --config configs/config_*mbrell*_4.txt --cluster_2D --render_mode

python run_nerf.py --config configs/config_*umping*_4.txt --cluster_2D

python run_nerf.py --config configs/config_*umping*_4.txt --cluster_2D --render_mode


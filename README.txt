Classical NE

This directory contains a small classical NEB implementation using the
Stillinger-Weber potential for Si. The final report uses four groups of tests:

1. SW pair-potential validation
2. SW triplet/angular validation
3. Fixed Si channel test
4. Vacancy-hop tests in diamond Si, with frozen and locally relaxed versions

Generated data is written to NEB_Data/, and generated figures are written to Plots/.

You can rebuild the fortran backend with make rebuild

The program can then be run for the tests with:

1. Pair Validation

Generate the two-atom energy scan:
python relax_main.py --initial-poscar Example_POSCARs/POSCAR_Si_line-init.vasp \
--final-poscar Example_POSCARs/POSCAR_Si_line-final.vasp --nimages 31 --neb-file NEB_Data/neb_pair.dat

Plot the neb curve and save it to Plots/
python plot_sw_pair_overlay.py


2. Triplet Validation

Generate the angular image scan and save the image geometries used to compute
the angle axis:
python relax_main.py --initial-poscar Example_POSCARs/POSCAR_Si_angle-init.vasp \
--final-poscar Example_POSCARs/POSCAR_Si_angle-final.vasp --nimages 31 --neb-file NEB_Data/neb_triplet.dat \
--write-images --images-dir NEB_Triplet_Images

Plot the neb curve and save it to Plots/
python plot_sw_triplet_overlay.py

3. Fixed Channel NEB

Run the fixed atom channel example:
python relax_main.py --initial-poscar Example_POSCARs/POSCAR_Si_channel-init.vasp \
--final-poscar Example_POSCARs/POSCAR_Si_channel-final.vasp --nimages 31 --neb-file NEB_Data/neb_channel.dat

Plot the neb curve and save it to Plots/
python plot_neb.py --neb-file NEB_Data/neb_channel.dat --out-file Plots/neb_channel.png --title "Fixed Channel NEB"


4. Vacancy Hop Version 1: Frozen Environment

Run the vacancy hop with only the hopping atom movable:

python relax_main.py --initial-poscar Example_POSCARs/POSCAR_Si_vacancy_A-init.vasp \
--final-poscar Example_POSCARs/POSCAR_Si_vacancy_A-final.vasp --pbc --nimages 31 --neb-file NEB_Data/neb_vacancy_A.dat

Plot the neb curve and save it to Plots/
python plot_neb.py --neb-file NEB_Data/neb_vacancy_A.dat --out-file Plots/neb_vacancy_A.png --title "Vacancy Hop 1"

5. Vacancy Hop Version 2: Locally Relaxed Environment

Run the vacancy hop with the hopping atom and nearby atoms movable:
--write-energies to output pair and triplet component profiles.

python relax_main.py --initial-poscar Example_POSCARs/POSCAR_Si_vacancy_B-init.vasp \
--final-poscar Example_POSCARs/POSCAR_Si_vacancy_B-final.vasp --pbc --nimages 61 --neb-file NEB_Data/neb_vacancy_B.dat \
--write-energies

Plot the three energy curves and save them to Plots/
python plot_neb.py --neb-file NEB_Data/neb_vacancy_B.dat --out-file Plots/neb_vacancy_B.png --title "Vacancy Hop 2"
python plot_neb.py --neb-file NEB_Data/neb_vacancy_B_pairE.dat --out-file Plots/neb_vacancy_B_pairE.png --title "Vacancy Hop 2 - Pair"
python plot_neb.py --neb-file NEB_Data/neb_vacancy_B_tripE.dat --out-file Plots/neb_vacancy_B_tripE.png --title "Vacancy Hop 2 - Triplet"
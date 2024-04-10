This is the code that I used to generate the DRM-ARM move count stats.

I use `g++ -Wall generate_drm_cases.cpp` to compile

and `./a.out` to generate the "raw_data.csv" file.

This file stores the bad corner/edge counts for DRM and ARM as well as the optimal path to the case and its length for each individual EO case.

Then running `python3 drm_arm_stats.py` will generate the csv files for average optimal, sub-6 and sub-7 chances and the best and worst algs for each DRM-ARM combo as a 2D table
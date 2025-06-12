# DRM-ARM++

These are visualizations of the probability than an EO configuration 
will have a sub-7 solution, for different combinations of  
"features" of the configuration. The features are easily computed by hand, including NISS prediction.

The features are:

| Feature             |Explanation|
|---------------------|--|
| n_pairs             |Number of bad corner/edge pairs|
| corner_arm          |Number of out-of-AR corners|
| edge_arm            |Number of out-of-AR edges|
| corner_orbit_split  |Number of bad corners in the HTR orbit with the fewest bad corners|
| corner_orbit_parity |0 = 4a corners, 1 = 4b corners|
| n_ppairs            |Number of top pseudo-pairs|
| n_spairs            |Number of side pairs|

The feature combinations are "optimal" in the sense that they were are the result of training a decision tree.
At each level, the training algorithm chooses the feature that does the best job of separating the 
sub-7 solutions from the 7-plus solutions.

Results:

## 0c
- [0c0e](0c0e_p_sub7.html)
- [0c2e](0c2e_p_sub7.html)
- [0c4e](0c4e_p_sub7.html)
- [0c6e](0c6e_p_sub7.html)
- [0c8e](0c8e_p_sub7.html)
## 2c
- [2c0e](2c0e_p_sub7.html)
- [2c2e](2c2e_p_sub7.html)
- [2c4e](2c4e_p_sub7.html)
- [2c6e](2c6e_p_sub7.html)
- [2c8e](2c8e_p_sub7.html)
## 3c
- [3c0e](3c0e_p_sub7.html)
- [3c2e](3c2e_p_sub7.html)
- [3c4e](3c4e_p_sub7.html)
- [3c6e](3c6e_p_sub7.html)
- [3c8e](3c8e_p_sub7.html)
## 4c
- [4c0e](4c0e_p_sub7.html)
- [4c2e](4c2e_p_sub7.html)
- [4c4e](4c4e_p_sub7.html)
- [4c6e](4c6e_p_sub7.html)
- [4c8e](4c8e_p_sub7.html)
## 5c
- [5c0e](5c0e_p_sub7.html)
- [5c2e](5c2e_p_sub7.html)
- [5c4e](5c4e_p_sub7.html)
- [5c6e](5c6e_p_sub7.html)
- [5c8e](5c8e_p_sub7.html)
## 6c
- [6c0e](6c0e_p_sub7.html)
- [6c2e](6c2e_p_sub7.html)
- [6c4e](6c4e_p_sub7.html)
- [6c6e](6c6e_p_sub7.html)
- [6c8e](6c8e_p_sub7.html)
## 7c
- [7c0e](7c0e_p_sub7.html)
- [7c2e](7c2e_p_sub7.html)
- [7c4e](7c4e_p_sub7.html)
- [7c6e](7c6e_p_sub7.html)
- [7c8e](7c8e_p_sub7.html)
## 8c
- [8c0e](8c0e_p_sub7.html)
- [8c2e](8c2e_p_sub7.html)
- [8c4e](8c4e_p_sub7.html)
- [8c6e](8c6e_p_sub7.html)
- [8c8e](8c8e_p_sub7.html)


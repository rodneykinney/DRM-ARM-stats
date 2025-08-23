# DRM-ARM++

These are visualizations of decision trees trained on the probability 
that a cube has a "findable" sub-7 EO-to-DR solution.

The trees use the following features:

| Feature Name        | Meaning                                             |
|---------------------|-----------------------------------------------------|
| n_pairs             | Number of top pairs (bad corner + bad edge)         |
| n_fake_pairs        | Number of top fake-pairs (mis-oriented corner+edge) |
| n_side_pairs        | Number of side pairs (white pair on F/B face)       |
| corner_arm          | Number of out-of-AR corners                         |
| edge_arm            | Number of out-of-AR edges                           |

At each level, the training algorithm chooses the feature that does the best job of separating the 
sub-7 solutions from the 7-plus solutions.

Trees by DRM:
- [4c4e](4c4e_p_sub7_findable.html)
- [3c2e](3c2e_p_sub7_findable.html)
- [7c8e](7c8e_p_sub7_findable.html)
- [4c2e](4c2e_p_sub7_findable.html)
- [5c4e](5c4e_p_sub7_findable.html)
- [5c6e](5c6e_p_sub7_findable.html)
- [4c6e](4c6e_p_sub7_findable.html)
- [6c6e](6c6e_p_sub7_findable.html)
- [3c4e](3c4e_p_sub7_findable.html)
- [2c2e](2c2e_p_sub7_findable.html)

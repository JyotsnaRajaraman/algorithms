# Minimum distance between points

Algorithm: Divide and conquer method to find minimum distance between points.

Special case: We want to find the minimum distance between pairs of clouds (each cloud may consist of a different number of points). The minimum distance between two points from the same cloud should not be considered.

Run time comparison:

(Naive vs divide and conquer) -> for 2 components with 3703, 615580 points each

Naive takes ~20 seconds while the new approach takes ~2.3 seconds

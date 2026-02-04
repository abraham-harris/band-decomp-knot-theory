# Deep Reinforcement Learning for Simplifying Braid Band Decompositions

## Brief Topological Background
In knot theory, a braid is a set of $n$ strings that are attached to a horizontal bar at the top and travel downward, crossing over and under each other however they like (without going upward) before attaching to a horizontal bar at the bottom.

![Braid](/README_images/braid.png "braid")

One reason braids are useful in knot theory is that they can be easily converted to knots or links by connecting the bottom strands to the top.

To more easily talk about braids, we can label their crossings. If the crossing occurs between strands $j$ and $k$ going from left to right, we label it $\sigma_j$. If the same crossing goes right to left, we label it $\sigma_{j-1}$.


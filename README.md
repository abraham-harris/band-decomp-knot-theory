# Deep Reinforcement Learning for Simplifying Braid Band Decompositions

## Brief Topological Background
In knot theory, a braid is a set of $n$ strings that are attached to a horizontal bar at the top and travel downward, crossing over and under each other however they like (without going upward) before attaching to a horizontal bar at the bottom.

<div align="center">
  <img src="/README_images/braid.png" alt="Braid" width="200">
  <br>
  <sub><em>Figure 1: an example of a braid.</em></sub>
  <br>
  <sub>Source: Dylan Skinner Blog (https://dylanskinner65.github.io/blog/braids.html)</sub>
  <br><br>
</div>

One reason braids are useful in knot theory is that they can be easily converted to knots or links by connecting the bottom strands to the top.

<div align="center">
  <img src="/README_images/braid_to_knot.png" alt="Braid" width="300">
  <br>
  <sub><em>Figure 2: turning a braid into a knot.</em></sub>
  <br>
  <sub>Source: Dylan Skinner Blog (https://dylanskinner65.github.io/blog/braids.html)</sub>
  <br><br>
</div>

To more easily talk about braids, we can label their crossings. If the crossing occurs between strands $j$ and $k$ going from left to right, we label it $\sigma_j$. If the same crossing goes right to left, we label it $\sigma_{j-1}$.

<div align="center">
  <img src="/README_images/crossings.png" alt="Braid" width="300">
  <br>
  <sub><em>Figure 3: crossings in a braid.</em></sub>
  <br>
  <sub>Source: Adams, Colin C. The Knot Book (page 133)</sub>
  <br><br>
</div>

This allows us to represent braids as “braid words.” For example, the braid in Figure 4 can be written as $\sigma_2 \sigma_1 \sigma_1 \sigma_2^{-1} \sigma_1 \sigma_1$.

<div align="center">
  <img src="/README_images/braid_word_example.png" alt="Braid" width="150">
  <br>
  <sub><em>Figure 4: braid example.</em></sub>
  <br>
  <sub>Source: Adams, Colin C. The Knot Book (page 133)</sub>
  <br><br>
</div>

Braids can also be decomposed into bands. A band is an element of the form $\omega \sigma_i \omega^{-1}$ where $\sigma_i$ is a crossing and $\omega$ is another braid in the braid group with the same number of strands. An upper bound on a braid’s minimal length band decomposition is the number of crossings in the braid, because each crossing is a trivial band.

## Research Goal
Finding a braid’s minimal length band decomposition can be challenging, since in some cases adding new crossings and twists to our braid can allow us to decompose it into shorter bands, even though the number of crossings has increased. I explore the possibility of training a deep reinforcement learning model to take as input a braid and output the shortest band decomposition it can find. This involves creating a custom RL environment, curriculum learning, and other techniques. 

Being able to find shortest band decompositions would be useful in studying other problems in knot theory, such as slice genus and quasipositive braid detection.

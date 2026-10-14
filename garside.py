"""Classical Garside left normal form for Artin braid words.

Simple factors are represented by permutations of the strand positions.
Generator indices are one-based, matching BandEnv's crossing words.
"""


def _tau(permutation):
    n = len(permutation)
    return tuple(n - 1 - permutation[n - 1 - j] for j in range(n))


def left_normal_form(n, word):
    """Return ``(power, factors)`` for Delta**power times simple factors."""
    identity = tuple(range(n))
    delta = tuple(range(n - 1, -1, -1))

    def swap_positions(permutation, index):
        permutation = list(permutation)
        permutation[index], permutation[index + 1] = (
            permutation[index + 1], permutation[index]
        )
        return tuple(permutation)

    def swap_values(permutation, index):
        return tuple(
            index + 1 if value == index else
            index if value == index + 1 else value
            for value in permutation
        )

    power, factors = 0, []
    for crossing in word:
        index = abs(crossing) - 1
        if crossing > 0:
            factors.append(swap_positions(identity, index))
        else:
            factors = [_tau(factor) for factor in factors]
            factors.append(swap_positions(delta, index))
            power -= 1

    changed = True
    while changed:
        changed = False
        for factor_index in range(len(factors) - 1):
            left, right = factors[factor_index:factor_index + 2]
            while True:
                positions = {value: position for position, value in enumerate(right)}
                index = next((index for index in range(n - 1)
                              if positions[index + 1] < positions[index]
                              and left[index] < left[index + 1]), None)
                if index is None:
                    break
                left = swap_positions(left, index)
                right = swap_values(right, index)
                changed = True
            factors[factor_index:factor_index + 2] = left, right

    while factors and factors[0] == delta:
        factors.pop(0)
        power += 1
    return power, [factor for factor in factors if factor != identity]

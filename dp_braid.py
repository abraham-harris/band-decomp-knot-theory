"""Meiners-style interval DP for a braid-word band-rank upper bound.

The free-group split/conjugate recurrence is augmented with exact braid
identity checks and the commuting-generator rule from Chapter 5 of
"Computing the Rank of Braids".  Its result is an upper bound, not an exact
braid rank in general.  Every returned band is an ordered word in signed
Artin generators, so ``decomposition`` has BandEnv's list-of-lists format.
"""

from dataclasses import dataclass

from garside import left_normal_form


def inverse(word):
    return tuple(-crossing for crossing in reversed(word))


def free_reduce(word):
    stack = []
    for crossing in word:
        if stack and stack[-1] == -crossing:
            stack.pop()
        else:
            stack.append(crossing)
    return tuple(stack)


def normal_form_key(braid_index, word):
    power, factors = left_normal_form(braid_index, word)
    return power, tuple(tuple(factor) for factor in factors)


@dataclass
class UpperBoundResult:
    upper_bound: int
    decomposition: list[list[int]]
    # Each pair certifies one band as conjugator * generator * inverse(conjugator).
    certificates: list[tuple[list[int], int]]
    reduced_word: list[int]


def validate_witness(word, braid_index, result):
    """Reject an invalid band count, band certificate, or braid product."""
    if len(result.decomposition) != result.upper_bound:
        raise ValueError("DP band count does not match its upper bound")
    if len(result.certificates) != len(result.decomposition):
        raise ValueError("DP band certificates do not match its decomposition")
    flat = []
    for band, (conjugator, generator) in zip(
            result.decomposition, result.certificates):
        if (not band or type(generator) is not int or generator == 0 or
                abs(generator) >= braid_index):
            raise ValueError("DP witness contains an invalid band generator")
        if any(type(crossing) is not int or crossing == 0 or
               abs(crossing) >= braid_index for crossing in band + conjugator):
            raise ValueError("DP witness contains an invalid crossing")
        certified_band = tuple(conjugator) + (generator,) + inverse(conjugator)
        if normal_form_key(braid_index, band) != normal_form_key(
                braid_index, certified_band):
            raise ValueError("DP witness contains a non-certified band")
        flat.extend(band)
    if normal_form_key(braid_index, flat) != normal_form_key(braid_index, word):
        raise ValueError("DP witness does not equal its input braid")
    return True


def braid_upper_bound(word, braid_index, *, verify=True):
    """Return a certified band decomposition and its braid-rank upper bound.

    The recurrence uses one longest valid conjugation per interval, matching
    the heuristic in Meiners's braid-group implementation.  The input is
    freely reduced first; this preserves its braid element.
    """
    if type(braid_index) is not int or braid_index < 2:
        raise ValueError("braid_index must be at least 2")
    word = tuple(word)
    if any(type(crossing) is not int or crossing == 0 or
           abs(crossing) >= braid_index for crossing in word):
        raise ValueError("word contains an invalid signed braid generator")
    reduced = free_reduce(word)
    length = len(reduced)
    if not length:
        result = UpperBoundResult(0, [], [], [])
        if verify:
            validate_witness(word, braid_index, result)
        return result

    forward = {}
    backward = {}
    identity = normal_form_key(braid_index, ())

    def interval_key(start, end):
        if start == end:
            return identity
        pair = start, end
        if pair not in forward:
            forward[pair] = normal_form_key(braid_index, reduced[start:end])
        return forward[pair]

    def inverse_interval_key(start, end):
        if start == end:
            return identity
        pair = start, end
        if pair not in backward:
            backward[pair] = normal_form_key(
                braid_index, inverse(reduced[start:end]))
        return backward[pair]

    def longest_conjugation(start, end):
        """Find prefix P and suffix S with P*S = 1, minimizing core length."""
        suffix_positions = {identity: end}
        chosen = None
        best_width = end - start
        for left_end in range(end - 1, start - 1, -1):
            suffix_positions[inverse_interval_key(left_end, end)] = left_end
            right_start = suffix_positions.get(interval_key(start, left_end))
            if right_start is None:
                continue
            width = right_start - left_end
            if ((left_end > start or right_start < end) and
                    (width < best_width or
                     (width == best_width and
                      (chosen is None or left_end < chosen[0])))):
                chosen = left_end, right_start
                best_width = width
        return chosen

    def commuting_bound(start, end):
        exponents = {}
        for crossing in reduced[start:end]:
            generator = abs(crossing)
            exponents[generator] = exponents.get(generator, 0) + (
                1 if crossing > 0 else -1)
        generators = sorted(exponents)
        if any(right - left < 2 for left, right in zip(
                generators, generators[1:])):
            return None
        return sum(abs(exponents[generator]) for generator in generators)

    scores = [[0] * (length + 1) for _ in range(length + 1)]
    choices = {}
    for span in range(1, length + 1):
        for start in range(length - span + 1):
            end = start + span
            if span == 1:
                scores[start][end] = 1
                choices[start, end] = ("letter",)
                continue

            best = span
            decision = ("split", start + 1)
            for cut in range(start + 1, end):
                candidate = scores[start][cut] + scores[cut][end]
                if candidate < best:
                    best, decision = candidate, ("split", cut)

            commuting = commuting_bound(start, end)
            if commuting is not None and commuting < best:
                best, decision = commuting, ("commute",)

            # Zero is optimal, and avoiding a braid search here matters for
            # words with many cancelling intervals.
            if best:
                conjugation = longest_conjugation(start, end)
                if conjugation is not None:
                    left_end, right_start = conjugation
                    candidate = scores[left_end][right_start]
                    if candidate < best:
                        best = candidate
                        decision = ("conjugate", left_end, right_start)

            scores[start][end] = best
            choices[start, end] = decision

    def reconstruct(start, end):
        if start == end:
            return []
        choice = choices[start, end]
        if choice[0] == "letter":
            return [((), reduced[start])]
        if choice[0] == "split":
            cut = choice[1]
            return reconstruct(start, cut) + reconstruct(cut, end)
        if choice[0] == "conjugate":
            left_end, right_start = choice[1:]
            prefix = reduced[start:left_end]
            return [(free_reduce(prefix + conjugator), generator)
                    for conjugator, generator in reconstruct(left_end, right_start)]
        exponents = {}
        for crossing in reduced[start:end]:
            generator = abs(crossing)
            exponents[generator] = exponents.get(generator, 0) + (
                1 if crossing > 0 else -1)
        return [((), generator if exponent > 0 else -generator)
                for generator, exponent in sorted(exponents.items())
                for _ in range(abs(exponent))]

    certificates = reconstruct(0, length)
    decomposition = [list(free_reduce(
        conjugator + (generator,) + inverse(conjugator)))
        for conjugator, generator in certificates]
    result = UpperBoundResult(
        scores[0][length], decomposition,
        [(list(conjugator), generator) for conjugator, generator in certificates],
        list(reduced),
    )
    if verify:
        validate_witness(word, braid_index, result)
    return result

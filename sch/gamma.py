import math
from functools import cache

@cache
def calculate_gammas(n: int, p: float):
    """
    Given maximal allowed probability of violation,
    compute gamma values for all integer k in range [0, n].

    Uses formulae (18) and (19) from the paper "The Price of Robustness".
    Note that gamma(k-1, p) <= gamma(k, p) <= k.
    This allows to compute all gammas in a single pass in O(n) time.

    Also note that bound from the paper is too relaxed for small k.
    For example, probability(k=1, g=1) = 0.5
                 probability(k=5, g=5) = 0.03125
    Although we know that probability(k, g=k) = 0.0.
    Therefore we initially set gamma(k, p) = k and then use formula (18)
    to improve this bound whenever possible.
    """
    gammas = [k for k in range(n+1)]
    for k in range(1, n+1):
        for g in range(gammas[k-1], k+1):
            if probability_given_n_gamma(k, g) <= p:
                gammas[k] = g
                break
    return gammas

def probability_given_n_gamma(n: int, gamma: float) -> float:
    def cal_c(nx: int, lx: int) -> float:
        if lx == 0 or lx == nx:
            return math.pow(0.5, nx)
        else:
            c = nx * math.log(nx / 2 / (nx - lx)) + lx * math.log((nx - lx) / lx)
            c = math.exp(c)
            c *= math.sqrt(nx / (nx - lx) / lx)
            c /= math.sqrt(2 * math.pi)
            return c

    nu = (n + gamma) / 2.0
    nu_f = math.floor(nu)
    mu = nu - nu_f

    formula_part1 = (1 - mu) * cal_c(n, nu_f)
    formula_part2 = sum([cal_c(n, ll) for ll in range(nu_f + 1, n + 1)])

    return formula_part1 + formula_part2

if __name__ == '__main__':
    gammas = calculate_gammas(2000, 0.01)
    assert gammas[5] == 5
    assert gammas[10] == 9
    assert gammas[100] == 25
    assert gammas[200] == 34
    assert gammas[2000] == 106
    assert probability_given_n_gamma(2000, 105) > 0.01


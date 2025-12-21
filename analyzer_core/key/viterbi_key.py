import numpy as np

def viterbi_log(log_pi, log_A, log_B):  
    """
    log_pi : (N,)        Inital Log-Likelihood
    log_A  : (N,N)       Transition Log-Likelihood
    log_B  : (T,N)       Emission Log-Likelihood
    return: path (T,)
    """
    T, N = log_B.shape[0], log_pi.shape[0]
    delta = np.full((T, N), -np.inf, dtype=np.float64)
    psi   = np.zeros((T, N), dtype=np.int32)

    delta[0] = log_pi + log_B[0]

    for t in range(1, T):
        scores = delta[t-1][:, None] + log_A
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(N)] + log_B[t]

    path = np.zeros(T, dtype=np.int32)
    path[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    return path

def calc_emission(chroma_subdiv, keyprob):
    ex_a = []
    for i in np.transpose(chroma_subdiv):
        ex_k = []
        for j in keyprob:
            ex_k.append(np.sum(np.multiply(j, i)))
        ex_a.append(ex_k)

    ex_a = np.transpose(ex_a)
    return ex_a

def to_log_B_from_scores(ex_a, temp=12.0, eps=1e-12):
    S = ex_a.T.astype(np.float64)
    S_scaled = S * temp
    m = np.max(S_scaled, axis=1, keepdims=True)
    P = np.exp(S_scaled - m)
    P /= (np.sum(P, axis=1, keepdims=True) + eps)
    return np.log(P + eps)

def build_transition_12(p_self= 0.99995, p_semi= 0.00004, p_fifth=0.000007, p_other=0.000003,eps=1e-12):
    N = 12
    probs = np.array([p_self, p_semi, p_fifth, p_other], dtype=float)
    s = probs.sum()
    if s <= 0:
        raise
    probs /= s
    p_self, p_semi, p_fifth, p_other = probs

    A = np.zeros((N, N), dtype=float)

    for s in range(N):
        # self state
        A[s, s] += p_self

        # one semitione
        A[s, (s + 1) % N] += p_semi / 2.0
        A[s, (s - 1) % N] += p_semi / 2.0

        # perfect fifth
        A[s, (s + 7) % N] += p_fifth / 2.0
        A[s, (s - 7) % N] += p_fifth / 2.0

        # others
        used = {s, (s+1)%N, (s-1)%N, (s+7)%N, (s-7)%N}
        others = [j for j in range(N) if j not in used]

        if p_other > 0 and others:
            tiny = p_other / len(others)
            for j in others:
                A[s, j] += tiny

        # normalize
        row_sum = A[s].sum()
        if row_sum <= 0:
            A[s] = 1.0 / N
        else:
            A[s] /= row_sum

    log_A = np.log(A + eps)
    return log_A


def build_log_pi(n = 24):
    pi = np.full(n, 1.0/n, dtype=np.float64)
    return np.log(pi + 1e-12)

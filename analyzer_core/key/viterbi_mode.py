import numpy as np

def _normalize_chroma_12T(chroma_12T: np.ndarray) -> np.ndarray:
    assert chroma_12T.shape[0] == 12, "expected chroma shape (12,T)"
    C = chroma_12T.T.astype(float)        # (T,12)
    C /= (C.sum(axis=1, keepdims=True) + 1e-12)
    return C

def _l1(v: np.ndarray) -> np.ndarray:
    s = float(v.sum())
    return (v / s) if s > 0 else v

TRIAD_M = np.array([0, 4, 7], dtype=int)   # Major triad
TRIAD_m = np.array([0, 3, 7], dtype=int)   # Minor triad

_DEGREE_OFFSET = {
    'I': 0, 'IV': 5, 'V': 7,
    'vi': 9, 'ii': 2, 'iii': 4
}

def _symbol_quality(symbol: str) -> str:
    if symbol in ('I', 'IV', 'V'):
        return 'M'
    if symbol in ('vi', 'ii', 'iii'):
        return 'm'
    raise ValueError(f"Unsupported symbol: {symbol}")

def _build_rotated_masks(symbol: str) -> np.ndarray:
    """
    symbol in 'I','IV','V','vi','ii','iii'
    """
    deg = _DEGREE_OFFSET[symbol]
    triad = TRIAD_M if _symbol_quality(symbol) == 'M' else TRIAD_m

    bank = np.zeros((12, 12), dtype=float)
    for root in range(12): 
        pcs = (root + deg + triad) % 12
        mask = np.zeros(12, dtype=float)
        mask[pcs] = 1.0
        bank[root] = _l1(mask)
    return bank  # shape (12,12)

def _pick_cols(A: np.ndarray, cols: np.ndarray) -> np.ndarray:
    T = A.shape[0]
    return A[np.arange(T), cols]

def emission_mode_logB(chroma_12T: np.ndarray, major_root_idx_series: np.ndarray, w_major=(1.0, 0.5, 0.5), w_relm =(1.0, 0.5, 0.5),):
    C = _normalize_chroma_12T(chroma_12T)
    T = C.shape[0]
    roots = np.asarray(major_root_idx_series, dtype=int)
    assert roots.shape == (T,), "major_root_idx_series must be shape (T,)"
    if np.any((roots < 0) | (roots > 11)):
        raise ValueError("major_root_idx_series must contain ints in [0,11].")
    needed_symbols = {
        "I":"I", "IV":"IV", "V":"V",
        "rel_i":"vi", "rel_iv":"ii", "rel_v":"iii"
    }
    masks = {k: _build_rotated_masks(sym) for k, sym in needed_symbols.items()}
    hits = {k: C @ masks[k].T for k in masks.keys()}

    I  = _pick_cols(hits["I"], roots);      IV  = _pick_cols(hits["IV"], roots);      V  = _pick_cols(hits["V"], roots)
    ri = _pick_cols(hits["rel_i"], roots);  riv = _pick_cols(hits["rel_iv"], roots);  rv = _pick_cols(hits["rel_v"], roots)

    M = w_major[0]*I + w_major[1]*IV + w_major[2]*V
    m = w_relm [0]*ri + w_relm [1]*riv + w_relm [2]*rv

    logB = np.log(np.stack([M, m], axis=1))  # (T,2)

    return logB

def build_mode_logA(self_bias=0.75):
    A = np.array([[self_bias,0],[0,self_bias]],dtype=float)
    m = A.max(axis=1, keepdims=True)
    ex = np.exp(A-m)
    return (A-m) - np.log(ex.sum(axis=1, keepdims=True))

def build_mode_logpi(init_bias=None):
    pi = np.array([0.5,0.5])
    if init_bias:
        if 'Major' in init_bias: pi[0]*=np.exp(init_bias['Major'])
        if 'Minor' in init_bias: pi[1]*=np.exp(init_bias['Minor'])
    pi /= pi.sum()
    return np.log(pi+1e-12)
"""Guide tree: BioNJ on pairwise times, rooted at the midpoint of the longest path."""
import numpy as np

MIN_T = 1e-3


def bionj(D):
    """BioNJ (Gascuel 1997). Unrooted adjacency {node: {nbr: length}}; leaves 0..n-1."""
    n = D.shape[0]
    size = 2 * n
    Dm = np.zeros((size, size)); Dm[:n, :n] = D
    V = Dm.copy()
    active = list(range(n))
    adj = {i: {} for i in range(n)}
    nxt = n
    while len(active) > 3:
        m = len(active)
        sub = Dm[np.ix_(active, active)]
        r = sub.sum(1)
        Qm = (m - 2) * sub - r[:, None] - r[None, :]
        np.fill_diagonal(Qm, np.inf)
        a, b = np.unravel_index(np.argmin(Qm), Qm.shape)
        i, j = active[a], active[b]
        li = 0.5 * Dm[i, j] + (r[a] - r[b]) / (2 * (m - 2))
        lj = Dm[i, j] - li
        others = [k for k in active if k not in (i, j)]
        lam = 0.5
        if V[i, j] > 0:
            lam = 0.5 + sum(V[j, k] - V[i, k] for k in others) / (2 * (m - 2) * V[i, j])
            lam = min(max(lam, 0.0), 1.0)
        u = nxt; nxt += 1
        for k in others:
            Dm[u, k] = Dm[k, u] = lam * (Dm[i, k] - li) + (1 - lam) * (Dm[j, k] - lj)
            V[u, k] = V[k, u] = lam * V[i, k] + (1 - lam) * V[j, k] - lam * (1 - lam) * V[i, j]
        adj[u] = {i: max(li, MIN_T), j: max(lj, MIN_T)}
        adj[i][u] = adj[u][i]; adj[j][u] = adj[u][j]
        active = others + [u]
    if len(active) == 3:
        x, y, z = active
        c = nxt
        adj[c] = {}
        for k, l in ((x, 0.5 * (Dm[x, y] + Dm[x, z] - Dm[y, z])),
                     (y, 0.5 * (Dm[x, y] + Dm[y, z] - Dm[x, z])),
                     (z, 0.5 * (Dm[x, z] + Dm[y, z] - Dm[x, y]))):
            adj[c][k] = adj[k][c] = max(l, MIN_T)
    elif len(active) == 2:
        x, y = active
        adj[x][y] = adj[y][x] = max(Dm[x, y], 2 * MIN_T)
    return adj


def _dists_from(adj, s):
    dist, prev, stack = {s: 0.0}, {s: None}, [s]
    while stack:
        u = stack.pop()
        for v, l in adj[u].items():
            if v not in dist:
                dist[v] = dist[u] + l; prev[v] = u; stack.append(v)
    return dist, prev


def midpoint_root(adj, n):
    """Rooted binary tree: {'root': r, 'children': {node: [(child, length), ...]}}."""
    if n == 1:
        return {"root": 0, "children": {0: []}}
    d0, _ = _dists_from(adj, 0)
    a = max(range(n), key=lambda k: d0[k])
    da, prev = _dists_from(adj, a)
    b = max(range(n), key=lambda k: da[k])
    half = da[b] / 2
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    path = path[::-1]                                  # a ... b
    for u, v in zip(path[:-1], path[1:]):
        if da[u] <= half <= da[v]:
            break
    root = max(adj) + 1
    lu = max(half - da[u], MIN_T / 2)
    lv = max(da[v] - half, MIN_T / 2)
    g = {x: dict(nb) for x, nb in adj.items()}
    del g[u][v], g[v][u]
    g[root] = {u: lu, v: lv}
    g[u][root] = lu; g[v][root] = lv
    children, stack, seen = {}, [root], {root}
    while stack:
        x = stack.pop()
        children[x] = []
        for y, l in g[x].items():
            if y not in seen:
                seen.add(y); children[x].append((y, l)); stack.append(y)
    return {"root": root, "children": children}


def guide_tree(D):
    return midpoint_root(bionj(np.asarray(D, float)), D.shape[0])


def postorder(tree):
    out, stack = [], [(tree["root"], False)]
    while stack:
        x, done = stack.pop()
        if done:
            out.append(x)
            continue
        stack.append((x, True))
        for c, _ in tree["children"][x]:
            stack.append((c, False))
    return out

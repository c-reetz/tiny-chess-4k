# Chess engine core, packed into main.py by pack.py.
# 0x88 board, pseudo-legal generation with king-capture detection,
# negamax alpha-beta + quiescence, PST eval, killers/history ordering,
# null-move pruning, iterative deepening under a fixed node budget.

P = "PNBRQK"
V = {"P": 100, "N": 325, "B": 335, "R": 500, "Q": 975, "K": 0}
N_OFF = (33, 31, 18, 14, -31, -33, -14, -18)
K_OFF = (16, 17, 1, -1, -16, -17, -15, 15)
B_OFF = (17, 15, -15, -17)
R_OFF = (16, 1, -1, -16)

PAWN = (
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0,
)
KNIGHT = (
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
)
BISHOP = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
)
ROOK = (
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
)
QUEEN = (
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
)
KING = (
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
)
PST = {"P": PAWN, "N": KNIGHT, "B": BISHOP, "R": ROOK, "Q": QUEEN, "K": KING}
MATE = 90000
BUDGET = 400000
KILL = {}
HIST = {}


class Budget(Exception):
    pass


SC = 0


def val(p, sq):
    w = p.isupper()
    i = ((sq >> 4) * 8 + (sq & 7)) if w else (((sq >> 4) * 8 + (sq & 7)) ^ 56)
    v = V[p.upper()] + PST[p.upper()][i]
    return v if w else -v


SQ = [i for i in range(128) if not i & 0x88]


def parse(fen):
    parts = fen.split()
    b = [0] * 128
    for i, row in enumerate(parts[0].split("/")):
        f = 0
        for c in row:
            if c.isdigit():
                f += int(c)
            else:
                b[(7 - i) * 16 + f] = c
                f += 1
    side = 0 if parts[1] == "w" else 1
    castle = parts[2] if len(parts) > 2 else "-"
    ep = None
    if len(parts) > 3 and parts[3] != "-":
        ep = (int(parts[3][1]) - 1) * 16 + "abcdefgh".index(parts[3][0])
    return b, side, castle, ep


def attacked(b, sq, by):
    for o in N_OFF:
        t = sq + o
        if not t & 0x88 and b[t] == ("N" if by == 0 else "n"):
            return True
    for o in K_OFF:
        t = sq + o
        if not t & 0x88 and b[t] == ("K" if by == 0 else "k"):
            return True
    for o in ((-15, -17) if by == 0 else (15, 17)):
        t = sq + o
        if not t & 0x88 and b[t] == ("P" if by == 0 else "p"):
            return True
    if by == 0:
        diag, orth = "BQ", "RQ"
    else:
        diag, orth = "bq", "rq"
    for offs, pieces in ((B_OFF, diag), (R_OFF, orth)):
        for o in offs:
            t = sq + o
            while not t & 0x88:
                p = b[t]
                if p:
                    if p in pieces:
                        return True
                    break
                t += o
    return False


def king_sq(b, side):
    k = "K" if side == 0 else "k"
    for i in SQ:
        if b[i] == k:
            return i
    return -1


def gen(b, side, castle, ep, caps_only=False):
    out = []
    mine = P if side == 0 else P.lower()
    theirs = P.lower() if side == 0 else P
    d = 16 if side == 0 else -16
    start = 1 if side == 0 else 6
    last = 7 if side == 0 else 0
    for i in SQ:
        p = b[i]
        if not p or p not in mine:
            continue
        u = p.upper()
        if u == "P":
            one = i + d
            if not one & 0x88 and not b[one] and not caps_only:
                if one >> 4 == last:
                    for q in "qrbn":
                        out.append((i, one, q))
                else:
                    out.append((i, one, ""))
                    two = one + d
                    if i >> 4 == start and not b[two]:
                        out.append((i, two, ""))
            for o in (d - 1, d + 1):
                t = i + o
                if t & 0x88:
                    continue
                if b[t] and b[t] in theirs or t == ep:
                    if t >> 4 == last:
                        for q in "qrbn":
                            out.append((i, t, q))
                    else:
                        out.append((i, t, ""))
        elif u == "N" or u == "K":
            for o in (N_OFF if u == "N" else K_OFF):
                t = i + o
                if t & 0x88 or (b[t] and b[t] in mine):
                    continue
                if caps_only and not b[t]:
                    continue
                out.append((i, t, ""))
        else:
            offs = B_OFF if u == "B" else R_OFF if u == "R" else K_OFF
            for o in offs:
                t = i + o
                while not t & 0x88:
                    q = b[t]
                    if q:
                        if q in theirs:
                            out.append((i, t, ""))
                        break
                    if not caps_only:
                        out.append((i, t, ""))
                    t += o
    if not caps_only and castle and castle != "-":
        base = 0 if side == 0 else 112
        kc, qc = ("K", "Q") if side == 0 else ("k", "q")
        if kc in castle and not b[base + 5] and not b[base + 6] \
                and b[base + 7] == ("R" if side == 0 else "r") \
                and not attacked(b, base + 4, 1 - side) \
                and not attacked(b, base + 5, 1 - side) \
                and not attacked(b, base + 6, 1 - side):
            out.append((base + 4, base + 6, ""))
        if qc in castle and not b[base + 3] and not b[base + 2] and not b[base + 1] \
                and b[base] == ("R" if side == 0 else "r") \
                and not attacked(b, base + 4, 1 - side) \
                and not attacked(b, base + 3, 1 - side) \
                and not attacked(b, base + 2, 1 - side):
            out.append((base + 4, base + 2, ""))
    return out


def make(b, m, castle, ep):
    global SC
    frm, to, promo = m
    p = b[frm]
    cap = b[to]
    oc = castle
    d = val(p, to) - val(p, frm)
    if cap:
        d -= val(cap, to)
    b[frm] = 0
    b[to] = promo.upper() if promo and p.isupper() else promo.lower() if promo else p
    if promo:
        d += val(b[to], to) - val(p, to)
    new_ep = None
    if p.upper() == "P":
        if to == ep and not cap and abs(to - frm) in (15, 17):
            cs = to - (16 if p.isupper() else -16)
            d -= val(b[cs], cs)
            b[cs] = 0
        elif abs(to - frm) == 32:
            new_ep = (frm + to) // 2
    if oc != "-":
        if p == "K":
            castle = castle.replace("K", "").replace("Q", "")
        elif p == "k":
            castle = castle.replace("k", "").replace("q", "")
        elif p == "R" and frm == 0:
            castle = castle.replace("Q", "")
        elif p == "R" and frm == 7:
            castle = castle.replace("K", "")
        elif p == "r" and frm == 112:
            castle = castle.replace("q", "")
        elif p == "r" and frm == 119:
            castle = castle.replace("k", "")
    if p.upper() == "K" and abs(to - frm) == 2:
        if to > frm:
            r0, r1 = frm + 3, frm + 1
        else:
            r0, r1 = frm - 4, frm - 1
        d += val(b[r0], r1) - val(b[r0], r0)
        b[r1] = b[r0]
        b[r0] = 0
    SC += d
    return (frm, to, promo, p, cap, oc, ep, castle, new_ep, d)


def unmake(b, undo):
    global SC
    frm, to, promo, p, cap, castle, ep = undo[:7]
    b[frm] = p
    b[to] = cap
    if p.upper() == "P" and to == ep and not cap and abs(to - frm) in (15, 17):
        b[to - (16 if p.isupper() else -16)] = "p" if p.isupper() else "P"
    if p.upper() == "K" and abs(to - frm) == 2:
        if to > frm:
            r0, r1 = frm + 3, frm + 1
        else:
            r0, r1 = frm - 4, frm - 1
        b[r0] = b[r1]
        b[r1] = 0
    SC -= undo[9]


def evaluate(b):
    return SC


def order(b, m, ply):
    cap = b[m[1]]
    if cap:
        return 100000 + V[cap.upper()] * 8 - V[b[m[0]].upper()]
    if m in KILL.get(ply, ()):
        return 90000
    return HIST.get((m[0], m[1]), 0)


def search(b, side, castle, ep, depth, alpha, beta, budget, ply=1):
    budget[0] -= 1
    if budget[0] <= 0:
        raise Budget
    ks = king_sq(b, side)
    check = attacked(b, ks, 1 - side)
    if depth <= 0:
        return quiesce(b, side, castle, ep, alpha, beta, budget, 1)
    if not check and depth >= 3 and abs(beta) < MATE - 1000:
        sc = -search(b, 1 - side, castle, None, depth - 3, -beta, -beta + 1,
                     budget, ply + 1)
        if sc >= beta:
            return beta
    best = -MATE * 2
    moves = gen(b, side, castle, ep)
    moves.sort(key=lambda m: order(b, m, ply), reverse=True)
    legal = 0
    for m in moves:
        cap = b[m[1]]
        if cap and cap.upper() == "K":
            return MATE - ply
        undo = make(b, m, castle, ep)
        if attacked(b, m[1] if m[0] == ks else ks, 1 - side):
            unmake(b, undo)
            continue
        legal += 1
        score = -search(b, 1 - side, undo[7], undo[8], depth - 1, -beta, -alpha,
                        budget, ply + 1)
        unmake(b, undo)
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if not cap:
                        ks = KILL.get(ply, ())
                        KILL[ply] = (m,) + tuple(x for x in ks[:1] if x != m)
                        k = (m[0], m[1])
                        HIST[k] = HIST.get(k, 0) + depth * depth
                    break
    if not legal:
        return -MATE + ply if check else 0
    return best


def quiesce(b, side, castle, ep, alpha, beta, budget, qd):
    budget[0] -= 1
    if budget[0] <= 0:
        raise Budget
    ks = king_sq(b, side)
    stand = evaluate(b)
    if side:
        stand = -stand
    if stand >= beta:
        return stand
    if stand > alpha:
        alpha = stand
    if qd > 4:
        return alpha
    moves = gen(b, side, castle, ep, True)
    moves.sort(key=lambda m: order(b, m, 99), reverse=True)
    for m in moves:
        cap = b[m[1]]
        if cap and cap.upper() == "K":
            return MATE - 1
        if stand + (V[cap.upper()] if cap else 100) + 200 < alpha:
            continue
        undo = make(b, m, castle, ep)
        if attacked(b, m[1] if m[0] == ks else ks, 1 - side):
            unmake(b, undo)
            continue
        score = -quiesce(b, 1 - side, undo[7], undo[8], -beta, -alpha, budget,
                         qd + 1)
        unmake(b, undo)
        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    return alpha


def uci_of(m):
    frm, to, promo = m
    return ("abcdefgh"[frm & 7] + str((frm >> 4) + 1)
            + "abcdefgh"[to & 7] + str((to >> 4) + 1) + promo)


def full(b):
    t = 0
    for i in SQ:
        if b[i]:
            t += val(b[i], i)
    return t


def best_move(fen, legal, budget=200000):
    global KILL, HIST
    KILL = {}
    HIST = {}
    b, side, castle, ep = parse(fen)
    have = {uci_of(m): m for m in gen(b, side, castle, ep)}
    pool = [have[u] for u in legal if u in have]
    if not pool:
        return legal[0]
    global SC
    SC = full(b)
    bunk = [budget]
    best = pool[0]
    for depth in range(1, 14):
        try:
            alpha, beta = -MATE * 2, MATE * 2
            scored = []
            for m in pool:
                undo = make(b, m, castle, ep)
                s = -search(b, 1 - side, undo[7], undo[8], depth - 1,
                            -beta, -alpha, bunk)
                unmake(b, undo)
                scored.append((s, m))
                if s > alpha:
                    alpha = s
            scored.sort(key=lambda x: -x[0])
            best = scored[0][1]
            pool = [m for _, m in scored]
        except Budget:
            break
    return uci_of(best)


if __name__ == "__main__":
    import sys, json, os, urllib.request as u
    E = os.environ.get("MAKEFUN_ENV_URL")
    if not E:
        def perft(b, side, castle, ep, k):
            if not k:
                return 1
            n2 = 0
            for mv in gen(b, side, castle, ep):
                un = make(b, mv, castle, ep)
                if attacked(b, king_sq(b, side), 1 - side):
                    unmake(b, un)
                    continue
                n2 += perft(b, 1 - side, un[7], un[8], k - 1)
                unmake(b, un)
            return n2

        F = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        for k in (1, 2, 3, 4):
            print("perft", k, perft(*parse(F), k))
        for fen, want in (
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
             (48, 2039, 97862)),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", (14, 191, 2812, 43238)),
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
             (6, 264, 9467, 422333)),
        ):
            got = tuple(perft(*parse(fen), k) for k in range(1, len(want) + 1))
            print("suite", "ok" if got == want else "BAD %s != %s" % (got, want))
    else:
        def W(path, body):
            r = u.Request(E + path, json.dumps(body).encode(),
                          {"content-type": "application/json"})
            try:
                return json.load(u.urlopen(r))
            except Exception:
                sys.exit()

        while 1:
            g = W("/game", {})
            if g["status"] == "done":
                break
            while g["status"] == "your_move":
                g = W("/move", {"game": g["game"],
                                "move": best_move(g["fen"], g["legal_moves"], BUDGET)})

import math, torch

def build_gta_controls_from_tokens(
    tokens,                     # e.g., ["W","W","W","W","A","W","R","R","W","A","W","W","W","W"]
    frames_per_token=4,         # match frame_seq_length (often 4)
    throttle_fwd=0.92,          # steady forward
    throttle_brake=0.25,        # “S”
    throttle_rev=-0.40,         # “R” if allowed (else we’ll clamp to 0)
    steer_mag=0.12,             # magnitude for A/D
    max_rate=0.03,              # per-frame steer rate limit
    allow_negative_throttle=False,  # set True only if your checkpoint used signed throttle
    device="cuda",
):
    """
    Returns tensor (T,2) where T = len(tokens) * frames_per_token
    Column0 = throttle, Column1 = steer
    """
    # symbolic → target (throttle, steer)
    def map_token(tok):
        t, s = 0.0, 0.0
        if tok == "W":
            t, s = throttle_fwd, 0.0
        elif tok == "A":
            t, s = throttle_fwd, -steer_mag
        elif tok == "D":
            t, s = throttle_fwd, +steer_mag
        elif tok == "S":
            t, s = throttle_brake, 0.0
        elif tok == "R":
            t = throttle_rev if allow_negative_throttle else 0.0
            s = 0.0
        else:
            # unknown token => coast
            t, s = 0.0, 0.0
        return t, s

    # Build per-token “targets”
    targets = [map_token(t) for t in tokens]

    # Expand to per-frame with rate-limited steer and exact per-token constancy
    T = len(tokens) * frames_per_token
    kb = torch.zeros((T, 2), dtype=torch.float32)

    last_steer = 0.0
    for i, (tgt_thr, tgt_str) in enumerate(targets):
        # throttle: piecewise-constant over the whole token
        for f in range(frames_per_token):
            t_idx = i*frames_per_token + f
            kb[t_idx, 0] = tgt_thr

            # steer: rate-limit toward token target
            delta = max(-max_rate, min(max_rate, tgt_str - last_steer))
            last_steer = last_steer + delta
            # clip steer to a safe band
            last_steer = max(-0.30, min(0.30, last_steer))
            kb[t_idx, 1] = last_steer

    # If negative throttle isn't allowed, clamp to [0,1]
    if not allow_negative_throttle:
        kb[:, 0].clamp_(0.0, 1.0)
    else:
        kb[:, 0].clamp_(-1.0, 1.0)

    return kb.to(device)

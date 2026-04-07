#!/usr/bin/env python3
"""
Generate a fancy Twitter GIF — Raven Memory Dynamics.
Black background, vivid slot colours, split prompt/reply ticker.
Output: ~/Desktop/raven_twitter.gif
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Constants ──────────────────────────────────────────────────────────────
LAYER, HEAD   = 23, 3
N             = 512
GEN_START     = 59
T             = 65
GEN_AMP       = 1.6            # moderate gen-phase amplification
ACTUAL_NEEDLE   = set(range(33, 41))   # ▁Rec all SS M 1 3 7 8
PASSWORD_TOKENS = {30, 31, 56, 57}    # ' pass','word' (both occurrences)

# Canvas
W, H    = 1200, 506
STEP_MS = 204
GAMMA   = 0.50     # higher = slower colour fill (0.20 = fast, 1.0 = linear)
LEVELS  = 12       # brightness quantisation steps

# ── Palette ────────────────────────────────────────────────────────────────
BG        = (  5,   7,  12)
BOX_BG    = (  9,  12,  20)
EDGE_COL  = ( 28,  48,  88)
TITLE_BG  = (  9,  11,  18)

SLOT_GRAY  = ( 22,  28,  42)

RED_DIM    = ( 90,   6,   6)   # dark red   → lerp to RED_FULL
BLU_DIM    = (  2,  12,  40)   # dark blue  → lerp to BLU_FULL
GRN_DIM    = (  2,  10,   5)   # dark green → lerp to GRN_FULL

RED_FULL   = (125,  12,  12)
BLU_FULL   = ( 45, 195, 255)
GRN_FULL   = ( 40, 230, 105)

ACCENT     = ( 90, 160, 255)
WHITE      = (240, 248, 255)

# Text — bright and legible
SUBTITLE   = (195, 220, 255)
LEGEND_TXT = (225, 240, 255)
PROGRESS_L = (170, 205, 255)
AXIS_RED   = (255, 100, 100)
AXIS_BLU   = (100, 195, 255)
AXIS_GRN   = ( 70, 240, 115)

PROMPT_LBL  = (180, 215, 255)
RAVEN_LBL = (255, 120, 120)
NEEDLE_CUR  = (255, 130, 130)
CTX_CUR     = (120, 255, 170)
NEEDLE_DON  = (255,  85,  85)
CTX_DON     = ( 55, 240, 140)
GEN_CUR     = (255, 160, 160)
GEN_DON     = (255, 120, 120)

# ── Display tokens (65) ───────────────────────────────────────────────────
DISPLAY_TOKENS = [
    '⟨s⟩',                                         # 0
    ' SS', 'Ms',                                   # 1-2
    ' of', 'ten',                                  # 3-4
    ' strug', 'gle',                               # 5-6
    ' on',                                         # 7
    ' rec', 'all',                                 # 8-9
    '-', 'he', 'avy',                              # 10-12
    ' tasks',                                      # 13
    ' when',                                       # 14
    ' mem', 'ory',                                 # 15-16
    ' up', 'dates',                               # 17-18
    ' stay', ' dense', ' and',                    # 19-21
    ' uni', 'form',                               # 22-23
    ' dur', 'ing',                                # 24-25
    ' train', 'ing',                              # 26-27
    '.',                                          # 28
    ' The',                                       # 29
    ' pass', 'word',                              # 30-31
    ' is',                                        # 32
    # 33-40 — password, aligns with ACTUAL_NEEDLE
    ' Rec', 'all', 'SS', 'M', '1', '3', '7', '8',
    '.',                                          # 41
    ' Ro', 'ut', 'ing',                           # 42-44
    ' stores', ' it', ' in',                      # 45-47
    ' ded', 'icated',                             # 48-49
    ' slots',                                     # 50
    '.',                                          # 51
    ' Now', ' tell', ' me',                       # 52-54
    ' the',                                       # 55
    ' pass', 'word',                              # 56-57
    ' =',                                         # 58
    # 59-64 — generation output
    ' Recall', 'SSM', ' 1', '3', '7', '8',
]
assert len(DISPLAY_TOKENS) == T

# ── Load data ──────────────────────────────────────────────────────────────
desk = os.path.expanduser('~/Desktop')
print("Loading routing data…")
routing = torch.load(f'{desk}/s_multihot_nslots512_topk256.pt', map_location='cpu')

rows = routing[LAYER][0, :, HEAD, :].numpy()   # [59, 512]

token_types = np.zeros(T, dtype=int)
for i in ACTUAL_NEEDLE:
    token_types[i] = 1
token_types[GEN_START:] = 1

needle_mean = rows[token_types[:GEN_START] == 1].mean(0)
gen_rows    = np.stack([needle_mean * GEN_AMP * (1.0 + 0.05 * k)
                        for k in range(T - GEN_START)])
all_rows    = np.vstack([rows, gen_rows])   # [65, 512]

# Cumulative frames
cumN = np.zeros(N); cumO = np.zeros(N)
framesN, framesO = [], []
for t in range(T):
    if token_types[t]:
        cumN += all_rows[t]
    else:
        cumO += all_rows[t]
    framesN.append(cumN.copy())
    framesO.append(cumO.copy())

# Sort by final needle fraction descending
final_tot = framesN[-1] + framesO[-1]
nfrac_end = np.divide(framesN[-1], final_tot,
                      out=np.zeros(N), where=final_tot > 0)
order = np.argsort(-nfrac_end)

fN = [f[order] for f in framesN]
fO = [f[order] for f in framesO]

# ── Dynamic thresholds for 25% red / 50% green / 25% blue ────────────────
nf_asc = np.sort(nfrac_end)            # ascending
GRN_THRESH = float(nf_asc[int(0.50 * N)])   # bottom 50% → green
RED_THRESH = float(nf_asc[int(0.75 * N)])   # top 25% → red
print(f"Thresholds:  green < {GRN_THRESH:.4f}  |  blue  |  red > {RED_THRESH:.4f}")
print(f"Slot counts: red={int((nfrac_end>RED_THRESH).sum())}  "
      f"blue={int(((nfrac_end>=GRN_THRESH)&(nfrac_end<=RED_THRESH)).sum())}  "
      f"green={int((nfrac_end<GRN_THRESH).sum())}")

# ── Pre-classify every slot once from its FINAL needle fraction ───────────
# Colour is fixed for all frames; only brightness changes over time.
# slot_type_ord[si]: 1=red  2=green  3=blue  (for sorted slot index si)
nfrac_sorted = nfrac_end[order]   # descending order (same as fN/fO)
slot_type_ord = np.where(nfrac_sorted > RED_THRESH, 1,
                np.where(nfrac_sorted < GRN_THRESH, 2, 3)).astype(int)

# Per-type brightness maxima:
#   red  slots are driven only by needle writes (fN)
#   green slots are driven only by context writes (fO)
#   blue  slots are driven by total writes (fN + fO)
final_N_s = fN[-1];  final_O_s = fO[-1]
red_mask = (slot_type_ord == 1)
grn_mask = (slot_type_ord == 2)
blu_mask = (slot_type_ord == 3)
maxN = max(float(final_N_s[red_mask].max()) if red_mask.any() else 1.0, 1e-9)
maxO = max(float(final_O_s[grn_mask].max()) if grn_mask.any() else 1.0, 1e-9)
maxB = max(float((final_N_s + final_O_s)[blu_mask].max()) if blu_mask.any() else 1.0, 1e-9)

# ── Memory display: shift by 1 so frame 0 = empty box ────────────────────
# Frame t shows the state BEFORE token t was processed.
# fN/fO[t] = state after token t, so we show fN/fO[t-1] at render step t.
# A leading zeros frame gives the empty-box first frame.
disp_fN = [np.zeros(N)] + fN   # index 0 = empty; index t = fN[t-1]
disp_fO = [np.zeros(N)] + fO

# ── Color helpers ──────────────────────────────────────────────────────────
def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

def slot_color(n_val, o_val, stype):
    """stype: 1=red  2=green  3=blue  (fixed from final classification).
    Red   slots driven by needle writes only  → invisible before first needle token.
    Green slots driven by context writes only → frozen during needle tokens.
    Blue  slots driven by all writes          → always active.
    """
    if stype == 1:                          # red — needle writes only
        if n_val < 1e-6:
            return SLOT_GRAY
        b = round((n_val / maxN) ** GAMMA * LEVELS) / LEVELS
        return lerp(RED_DIM, RED_FULL, b)
    elif stype == 2:                        # green — context writes only
        if o_val < 1e-6:
            return SLOT_GRAY
        b = round((o_val / maxO) ** GAMMA * LEVELS) / LEVELS
        return lerp(GRN_DIM, GRN_FULL, b)
    else:                                   # blue — any writes
        tot = n_val + o_val
        if tot < 1e-6:
            return SLOT_GRAY
        b = round((tot / maxB) ** GAMMA * LEVELS) / LEVELS
        return lerp(BLU_DIM, BLU_FULL, b)

def brighten(c, f=1.6):
    return tuple(min(255, int(x * f)) for x in c)

# ── Fonts ──────────────────────────────────────────────────────────────────
SF   = '/System/Library/Fonts/SFNS.ttf'
MONO = '/System/Library/Fonts/SFNSMono.ttf'
HEL  = '/System/Library/Fonts/HelveticaNeue.ttc'

def F(path, size):
    try:   return ImageFont.truetype(path, size)
    except:
        try: return ImageFont.truetype(HEL, size)
        except: return ImageFont.load_default()

f_title  = F(SF,   27)
f_sub    = F(SF,   15)
f_label  = F(SF,   15)
f_phase  = F(MONO, 14)
f_tok    = F(MONO, 12)
f_tok_lg = F(MONO, 15)

# ── Layout ─────────────────────────────────────────────────────────────────
PAD     = 22
HDR_H   = 62
BOX_Y   = HDR_H + 12
BOX_H   = 195
BOX_X0  = PAD
BOX_X1  = W - PAD
BOX_W   = BOX_X1 - BOX_X0
SLOT_W  = BOX_W / N
AX_Y    = BOX_Y + BOX_H + 5
TK_Y    = AX_Y + 28
TK_H    = 158
LG_Y    = TK_Y + TK_H + 8

TK_MID    = TK_Y + TK_H // 2 + 4
PROMPT_Y0 = TK_Y + 8
RAVEN_Y0 = TK_MID + 10


# ── Render one frame ───────────────────────────────────────────────────────
def render(t):
    img = Image.new('RGB', (W, H), BG)
    d   = ImageDraw.Draw(img)

    # Memory state: show pre-token-t state so frame 0 is empty
    vN = disp_fN[t]
    vO = disp_fO[t]

    # ── Header ────────────────────────────────────────────────────────────
    d.rectangle([0, 0, W, HDR_H], fill=TITLE_BG)
    d.line([0, HDR_H, W, HDR_H], fill=(18, 30, 58), width=1)

    d.text((PAD, 12), 'RAVEN', fill=ACCENT, font=f_title)
    aft = d.textbbox((PAD, 12), 'RAVEN', font=f_title)[2]
    d.text((aft + 2, 12), '  MEMORY DYNAMICS', fill=WHITE, font=f_title)

    sub = f'Layer {LAYER}  ·  Head {HEAD+1}  ·  512 memory slots  ·  step {t+1} / {T}'
    d.text((PAD, 43), sub, fill=SUBTITLE, font=f_sub)

    pb_x0 = W - PAD - 200
    d.rectangle([pb_x0, 30, pb_x0 + 200, 37], fill=(15, 22, 38))
    fw = int(200 * (t + 1) / T)
    if fw:
        d.rectangle([pb_x0, 30, pb_x0 + fw, 37], fill=ACCENT)
    d.text((pb_x0, 40), 'progress', fill=PROGRESS_L, font=f_label)

    # ── Memory box ────────────────────────────────────────────────────────
    d.rectangle([BOX_X0, BOX_Y, BOX_X1, BOX_Y + BOX_H], fill=BOX_BG)

    for si in range(N):
        c  = slot_color(vN[si], vO[si], slot_type_ord[si])
        x0 = BOX_X0 + si * SLOT_W
        x1 = x0 + max(SLOT_W - 0.2, 1.0)
        d.rectangle([x0, BOX_Y, x1, BOX_Y + BOX_H], fill=c)

    # Flash: most-active slots at this step
    if t > 0:
        cur_ord = all_rows[t - 1][order]   # same -1 shift
        for si in np.argsort(-cur_ord)[:4]:
            base  = slot_color(vN[si], vO[si], slot_type_ord[si])
            flash = brighten(base, 1.7)
            x0 = BOX_X0 + si * SLOT_W
            d.rectangle([x0, BOX_Y, x0 + max(SLOT_W, 1.0), BOX_Y + 6], fill=flash)

    # Border + corner accents
    d.rectangle([BOX_X0, BOX_Y, BOX_X1, BOX_Y + BOX_H], outline=EDGE_COL, width=1)
    CL = 14
    for cx, cy, dx, dy in [
        (BOX_X0, BOX_Y,          1,  1),
        (BOX_X1, BOX_Y,         -1,  1),
        (BOX_X0, BOX_Y + BOX_H,  1, -1),
        (BOX_X1, BOX_Y + BOX_H, -1, -1),
    ]:
        d.line([(cx, cy), (cx + dx * CL, cy)],    fill=ACCENT, width=2)
        d.line([(cx, cy), (cx, cy + dy * CL)],    fill=ACCENT, width=2)

    # ── Axis labels ───────────────────────────────────────────────────────
    d.text((BOX_X0, AX_Y), '← retrieval',  fill=AXIS_RED, font=f_label)
    d.text((W//2,   AX_Y), 'shared',        fill=AXIS_BLU, font=f_label, anchor='ma')
    d.text((BOX_X1, AX_Y), 'context →',    fill=AXIS_GRN, font=f_label, anchor='ra')

    # ── Token ticker (split layout) ───────────────────────────────────────
    d.rectangle([PAD, TK_Y, W - PAD, TK_Y + TK_H],
                fill=(8, 12, 20), outline=(25, 42, 75))
    d.line([PAD + 6, TK_MID, W - PAD - 6, TK_MID], fill=(25, 42, 75), width=1)

    # ── Prompt half ──────────────────────────────────────────────────────
    lx = PAD + 10
    ly = PROMPT_Y0
    d.text((lx, ly), 'Prompt:', fill=PROMPT_LBL, font=f_phase)
    tx = lx + 76

    for i in range(min(t + 1, GEN_START)):
        tok    = DISPLAY_TOKENS[i]
        needle = i in ACTUAL_NEEDLE or i in PASSWORD_TOKENS
        is_cur = (i == t)

        col = (NEEDLE_CUR if needle else CTX_CUR) if is_cur else \
              (NEEDLE_DON if needle else CTX_DON)

        d.text((tx, ly), tok, fill=col, font=f_tok)
        tx += d.textbbox((tx, ly), tok, font=f_tok)[2] - tx

        if tx > W - PAD - 70:
            tx  = lx + 76
            ly += 17
            if ly > TK_MID - 18:
                break

    # ── Raven half ───────────────────────────────────────────────────────
    gx = PAD + 10
    gy = RAVEN_Y0
    d.text((gx, gy), 'Raven:', fill=RAVEN_LBL, font=f_phase)
    gx += 82

    if t >= GEN_START:
        for i in range(GEN_START, min(t + 1, T)):
            tok    = DISPLAY_TOKENS[i]
            is_cur = (i == t)
            col    = GEN_CUR if is_cur else GEN_DON
            d.text((gx, gy), tok, fill=col, font=f_tok_lg)
            gx += d.textbbox((gx, gy), tok, font=f_tok_lg)[2] - gx
    else:
        if (t % 4) < 2:
            d.text((gx, gy), '▋', fill=(55, 75, 115), font=f_tok_lg)

    # ── Legend ────────────────────────────────────────────────────────────
    items = [
        (RED_FULL,  'Retrieval slots'),
        (BLU_FULL,  'Shared slots'),
        (GRN_FULL,  'Context slots'),
        (SLOT_GRAY, 'Unwritten'),
    ]
    lx = PAD
    for color, label in items:
        d.rectangle([lx, LG_Y + 2, lx + 16, LG_Y + 16], fill=color)
        d.text((lx + 22, LG_Y - 1), label, fill=LEGEND_TXT, font=f_label)
        lx += 175

    return img


# ── Render & save ──────────────────────────────────────────────────────────
print(f"Rendering {T} frames at {W}×{H}…")
frames = []
for t in range(T):
    print(f"  frame {t+1:2d}/{T}", end='\r', flush=True)
    frames.append(render(t))
print()

out = os.path.expanduser('~/Desktop/raven_twitter.gif')
print(f"Saving → {out}")
frames[0].save(
    out,
    save_all=True,
    append_images=frames[1:],
    duration=STEP_MS,
    loop=0,
    optimize=True,
)
sz = os.path.getsize(out) / 1e6
print(f"✓ Done  —  {sz:.1f} MB   ({W}×{H}, {T} frames, {STEP_MS}ms/frame)")

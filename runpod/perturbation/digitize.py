import numpy as np
from PIL import Image
im = np.array(Image.open('wiki_img/ps_i20.png').convert('RGB')).astype(int)
H, W, _ = im.shape
# axes frame: find the black rectangle (plot area) -> x 105..720, y 73..533 ; calibrate from gridlines (light grey)
def near(c, tol): return (np.abs(im - np.array(c)).sum(axis=2) < tol)
blue  = near((31,119,180), 40)
green = near((44,160,44), 60)
# calibration: x: 10^1 at 170, 10^3 at 586 ; y: 10^0 at 73, 10^-10 at 533 (tick label positions in the figure)
x0, x1 = 170.0, 586.0
y0, y1 = 73.0, 533.0
def k_of(px): return 10**(1 + (px - x0) / (x1 - x0) * 2)
def p_of(py): return 10**(0 - (py - y0) / (y1 - y0) * 10)
for name, mask in (('blue_base1024', blue), ('green_hybrid', green)):
    mask[:, :110] = mask[:, :110]   # keep
    mask[480:, :510] = False        # exclude legend box (x<505,y>438)
    mask[438:, :505] = False
    cols = np.where(mask.any(axis=0))[0]
    out = []
    for x in cols:
        ys = np.where(mask[:, x])[0]
        out.append((k_of(x), p_of(ys.mean())))
    out = np.array(out)
    np.savetxt(name + '.txt', out, header='k P(k) digitized from wiki image (+-~5%% in P)')
    print(name, len(out), out[::40].round(6).tolist())

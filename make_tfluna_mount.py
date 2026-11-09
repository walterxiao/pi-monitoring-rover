#!/usr/bin/env python3
"""Generate tf-luna mount STL: a glue-on locating tray for the Benewake
TF-Luna LiDAR (35 x 21.25 x 12.5 mm) on the Pi rover chassis.

The sensor sits in the tray with its lenses facing forward (-Y).
A dab of hot glue or double-sided tape under the sensor fixes it;
glue the tray's flat base to the chassis. No supports needed.

All dimensions in mm. Pure stdlib; writes a binary STL.
"""
import struct

# Sensor: 35 long (Y) x 21.25 wide (X) x 12.5 tall (Z).
# Tray pocket: 36 (Y) x 30 (X) — 1 mm slack front/back, room for the
# sensor's side mounting ears; front lip + back posts locate it.
BOXES = [
    # base plate: 34 x 40 x 3
    (-17, -20, 0, 17, 20, 3),
    # left rail: 2 thick, rises 4 mm above the base top
    (-17, -20, 2, -15, 20, 7),
    # right rail
    (15, -20, 2, 17, 20, 7),
    # front lip: 2.5 mm tall, stops the sensor sliding forward,
    # low enough to stay below the lenses
    (-15, -20, 2, 15, -18, 5.5),
    # back posts (two, with a 12 mm gap in the middle for the cable)
    (-15, 18, 2, -6, 20, 7),
    (6, 18, 2, 15, 20, 7),
]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _tri(p1, p2, p3, want):
    """One triangle; flips winding if its normal is not `want`."""
    n = _cross(_sub(p2, p1), _sub(p3, p1))
    if (n[0] * want[0] + n[1] * want[1] + n[2] * want[2]) < 0:
        p2, p3 = p3, p2
        n = _cross(_sub(p2, p1), _sub(p3, p1))
    ln = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5
    n = (n[0] / ln, n[1] / ln, n[2] / ln)
    return (n, p1, p2, p3)


def box_tris(x0, y0, z0, x1, y1, z1):
    v = {
        "000": (x0, y0, z0), "100": (x1, y0, z0),
        "110": (x1, y1, z0), "010": (x0, y1, z0),
        "001": (x0, y0, z1), "101": (x1, y0, z1),
        "111": (x1, y1, z1), "011": (x0, y1, z1),
    }
    faces = [
        ((0, 0, -1), "000", "100", "110", "010"),  # bottom
        ((0, 0, 1), "001", "011", "111", "101"),   # top
        ((-1, 0, 0), "000", "010", "011", "001"),  # -X
        ((1, 0, 0), "100", "101", "111", "110"),   # +X
        ((0, -1, 0), "000", "001", "101", "100"),  # -Y (front)
        ((0, 1, 0), "010", "110", "111", "011"),   # +Y (back)
    ]
    tris = []
    for want, a, b, c, d in faces:
        tris.append(_tri(v[a], v[b], v[c], want))
        tris.append(_tri(v[a], v[c], v[d], want))
    return tris


def main():
    tris = []
    for b in BOXES:
        tris.extend(box_tris(*b))
    out = b"\x00" * 80 + struct.pack("<I", len(tris))
    for n, p1, p2, p3 in tris:
        out += struct.pack("<12fH", n[0], n[1], n[2],
                           p1[0], p1[1], p1[2],
                           p2[0], p2[1], p2[2],
                           p3[0], p3[1], p3[2], 0)
    path = "/home/hatch/workspace/pi-monitoring-rover/tfluna_mount.stl"
    with open(path, "wb") as f:
        f.write(out)
    xs = [p[i] for _, p1, p2, p3 in tris for p in (p1, p2, p3) for i in (0,)]
    ys = [p[1] for _, p1, p2, p3 in tris for p in (p1, p2, p3)]
    zs = [p[2] for _, p1, p2, p3 in tris for p in (p1, p2, p3)]
    print(f"wrote {path}: {len(tris)} triangles")
    print(f"bbox X {min(xs)}..{max(xs)}  Y {min(ys)}..{max(ys)}  "
          f"Z {min(zs)}..{max(zs)} mm")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""TF-Luna holder for the Pi monitoring rover (4WD acrylic chassis kit). v2.

Mounts on the front nose of the upper deck (battery side), sensor facing
forward, beam horizontal. Bolted through the chassis slots with M3 hardware
(slots cross the chassis slots, so exact hole positions don't matter), or
taped down as a fallback.

Sensor facts (Benewake TF-Luna dimension drawing, verified Nov 13 2026):
  overall 35 wide x 21.25 tall x 13.5 deep (front lens face to back)
  main body 26.5 wide x 19 tall x 12 deep
  2 lenses on the wide front face, dia 10.5, centers 10.5 apart,
    lens centers 9 mm below the top edge of the front face
  2 mounting ears at the sides, mid-height, 2 mm thick, 2x dia 2.2 holes,
    hole centers 30.5 mm apart (ears reach the full 35 mm width)
  cable exits the BOTTOM face, centered, via an 18 mm wide tab

Holder: the sensor slides in from the front; its ears ride in horizontal
channels in the side walls, the bottom tab seats into the cable cutout, and
the cable exits rearward through the notch in the back plate.

v2 polish: rounded base + top-plate corners, ramped detents, lead-in
chamfers on the pocket floor and ear-channel mouths, beveled top-plate
front edge (helps the bridge print cleanly).

Units: mm. Print flat on the bed, PLA, 0.2 mm layers, no supports.
"""
import numpy as np
import trimesh
from trimesh.boolean import union, difference

# ---------------------------------------------------------------- parameters
BASE_W, BASE_D, BASE_T = 64.0, 36.0, 3.0      # base plate (X, Y, Z)
BASE_CY = 10.0                                 # base Y center -> Y in [-8, 28]
BASE_R = 4.0                                   # base corner radius
POCKET_W = 27.4                                # inner width between side walls
WALL_T = 3.3                                   # side wall thickness
WALL_X0 = POCKET_W / 2                         # inner wall face
WALL_X1 = WALL_X0 + WALL_T                     # outer wall face
POCKET_Y0, POCKET_Y1 = 0.0, 14.0               # pocket front / back-plate face
BACK_T = 2.5
FLOOR_Z0, FLOOR_Z1 = 10.0, 13.0                # pocket floor (sensor sits on Z1)
TOP_Z0, TOP_Z1 = 37.0, 40.0                    # top plate
TOP_R = 3.0                                    # top plate corner radius
CHAN_Z0, CHAN_Z1 = 22.7, 29.5                  # ear channels in side walls
SLOT_W = 3.4                                   # M3 mounting slots

# ------------------------------------------------------------------- helpers
def box(sx, sy, sz, cx, cy, cz):
    m = trimesh.creation.box(extents=[sx, sy, sz])
    m.apply_translation([cx, cy, cz])
    return m

def cyl_z(r, h, cx, cy, cz, sections=32):
    m = trimesh.creation.cylinder(radius=r, height=h, sections=sections)
    m.apply_translation([cx, cy, cz])
    return m

def rounded_rect_prism(w, d, h, r, cx, cy, cz):
    """Prism with rounded footprint corners (radius r)."""
    parts = [box(w - 2 * r, d, h, cx, cy, cz),
             box(w, d - 2 * r, h, cx, cy, cz)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(cyl_z(r, h, cx + sx * (w / 2 - r),
                               cy + sy * (d / 2 - r), cz))
    return union(parts, engine="manifold")

def rot_x_cut(sx, sy, sz, cx, cy, cz, deg=45.0):
    """Box rotated about the X axis through its center (chamfer cutter)."""
    m = box(sx, sy, sz, cx, cy, cz)
    m.apply_transform(trimesh.transformations.rotation_matrix(
        np.radians(deg), [1, 0, 0], [cx, cy, cz]))
    return m

def capsule_x(length, width, cx, cy, z0, z1):
    """Capsule slot along X, cut through Z0..Z1."""
    straight = length - width
    return union([
        box(straight, width, z1 - z0, cx, cy, (z0 + z1) / 2),
        cyl_z(width / 2, z1 - z0, cx - straight / 2, cy, (z0 + z1) / 2),
        cyl_z(width / 2, z1 - z0, cx + straight / 2, cy, (z0 + z1) / 2),
    ], engine="manifold")

def capsule_y(length, width, cx, cy, z0, z1):
    """Capsule slot along Y, cut through Z0..Z1."""
    straight = length - width
    return union([
        box(width, straight, z1 - z0, cx, cy, (z0 + z1) / 2),
        cyl_z(width / 2, z1 - z0, cx, cy - straight / 2, (z0 + z1) / 2),
        cyl_z(width / 2, z1 - z0, cx, cy + straight / 2, (z0 + z1) / 2),
    ], engine="manifold")

# ------------------------------------------------------------- positive parts
parts = []
# base plate, rounded corners
parts.append(rounded_rect_prism(BASE_W, BASE_D, BASE_T, BASE_R,
                                0, BASE_CY, BASE_T / 2))
# back plate (full height, ties walls + top together)
parts.append(box(2 * WALL_X1, BACK_T, TOP_Z1 - BASE_T,
                 0, POCKET_Y1 + BACK_T / 2, (BASE_T + TOP_Z1) / 2))
# side walls, lower (base -> ear channel) and upper (channel -> top)
for s in (-1, 1):
    cx = s * (WALL_X0 + WALL_T / 2)
    parts.append(box(WALL_T, POCKET_Y1 - POCKET_Y0, CHAN_Z0 - BASE_T,
                     cx, (POCKET_Y0 + POCKET_Y1) / 2, (BASE_T + CHAN_Z0) / 2))
    parts.append(box(WALL_T, POCKET_Y1 - POCKET_Y0, TOP_Z1 - CHAN_Z1,
                     cx, (POCKET_Y0 + POCKET_Y1) / 2, (CHAN_Z1 + TOP_Z1) / 2))
# pocket floor (sensor sits on top of this; cable space underneath)
parts.append(box(POCKET_W, POCKET_Y1 - POCKET_Y0, FLOOR_Z1 - FLOOR_Z0,
                 0, (POCKET_Y0 + POCKET_Y1) / 2, (FLOOR_Z0 + FLOOR_Z1) / 2))
# top plate, rounded corners
parts.append(rounded_rect_prism(2 * WALL_X1, POCKET_Y1 + BACK_T - POCKET_Y0,
                                TOP_Z1 - TOP_Z0, TOP_R,
                                0, (POCKET_Y0 + POCKET_Y1 + BACK_T) / 2,
                                (TOP_Z0 + TOP_Z1) / 2))
# ramped detents: 0 protrusion at the front, 0.7 mm at 2 mm in (smooth snap)
try:
    from shapely.geometry import Polygon
    for s in (-1, 1):
        tri = Polygon([(s * WALL_X0, 0.0),
                       (s * (WALL_X0 - 0.7), 2.0),
                       (s * WALL_X0, 2.0)])
        ramp = trimesh.creation.extrude_polygon(tri, height=CHAN_Z0 - FLOOR_Z1)
        ramp.apply_translation([0, 0, FLOOR_Z1])
        parts.append(ramp)
except ImportError:
    for s in (-1, 1):
        parts.append(box(0.7, 2.0, CHAN_Z0 - FLOOR_Z1,
                         s * (WALL_X0 - 0.35), POCKET_Y0 + 1.0,
                         (FLOOR_Z1 + CHAN_Z0) / 2))

body = union(parts, engine="manifold")

# ------------------------------------------------------------------- cutouts
cuts = []
# 4x M3 mounting slots in the base (cross the chassis slots)
cuts.append(capsule_x(44, SLOT_W, 0, -4.0, -0.5, BASE_T + 0.5))
cuts.append(capsule_x(44, SLOT_W, 0, 22.0, -0.5, BASE_T + 0.5))
cuts.append(capsule_y(20, SLOT_W, -25.0, 10.0, -0.5, BASE_T + 0.5))
cuts.append(capsule_y(20, SLOT_W, 25.0, 10.0, -0.5, BASE_T + 0.5))
# cable cutout in the pocket floor (bottom tab seats here, cable drops through)
cuts.append(box(20.0, 11.0, FLOOR_Z1 - FLOOR_Z0 + 0.4,
                0, 7.5, (FLOOR_Z0 + FLOOR_Z1) / 2))
# cable exit notch in the back plate (rearward exit at deck level)
cuts.append(box(16.0, BACK_T + 0.4, 7.4, 0, POCKET_Y1 + BACK_T / 2, 6.5))
# lead-in chamfer on the pocket floor front edge (sensor slides up onto it)
cuts.append(rot_x_cut(27.0, 4.0, 4.0, 0, 0.0, FLOOR_Z1))
# funnel chamfers at the ear-channel mouths (ears find the channels)
for s in (-1, 1):
    cx = s * (WALL_X0 + WALL_T / 2)
    cuts.append(rot_x_cut(3.5, 3.0, 3.0, cx, 0.0, CHAN_Z0))
    cuts.append(rot_x_cut(3.5, 3.0, 3.0, cx, 0.0, CHAN_Z1))
# bevel under the top plate front edge (cleaner bridge, finished look)
cuts.append(rot_x_cut(30.0, 4.0, 4.0, 0, 0.0, TOP_Z0))

result = difference([body, union(cuts, engine="manifold")], engine="manifold")
result.remove_unreferenced_vertices()
result.merge_vertices()

# ------------------------------------------------------------------ validate
print("watertight:", result.is_watertight)
print("volume mm^3:", round(result.volume, 1))
print("bounds min:", np.round(result.bounds[0], 2))
print("bounds max:", np.round(result.bounds[1], 2))
print("triangles:", len(result.faces))
print("euler number:", result.euler_number)

out_stl = "/home/hatch/workspace/goals/mobile-pi-monitoring-rover/files/tfluna_holder.stl"
result.export(out_stl)
print("wrote", out_stl)

# ------------------------------------------------------------------- preview
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def sensor_ghost():
    """Translucent sensor model at its seated position (front face at Y=0.5)."""
    gs = []
    gs.append(box(26.5, 12.0, 19.0, 0, 6.5, FLOOR_Z1 + 9.5))          # main body
    for s in (-1, 1):                                                 # ears
        gs.append(box(4.25, 2.0, 4.5, s * 15.375, 1.5, FLOOR_Z1 + 12.25))
    gs.append(box(18.0, 6.8, 2.25, 0, 6.75, FLOOR_Z1 - 1.125))        # bottom tab
    g = union(gs, engine="manifold")
    return g

def draw(ax, mesh, color, alpha, elev, azim, title):
    tris = mesh.triangles
    ax.add_collection3d(Poly3DCollection(tris, facecolor=color,
                                         edgecolor=(0.2, 0.2, 0.2, 0.25),
                                         linewidths=0.2, alpha=alpha))
    ax.set_xlim(-34, 34); ax.set_ylim(-10, 30); ax.set_zlim(0, 42)
    ax.set_box_aspect((68, 40, 42))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()

ghost = sensor_ghost()
fig = plt.figure(figsize=(13, 4.6))
ax1 = fig.add_subplot(131, projection="3d")
draw(ax1, result, (0.55, 0.75, 0.95), 1.0, 22, -58, "Holder (isometric)")
ax2 = fig.add_subplot(132, projection="3d")
draw(ax2, result, (0.55, 0.75, 0.95), 0.55, 6, -90, "Front — sensor seated (orange)")
draw(ax2, ghost, (1.0, 0.55, 0.15), 0.75, 6, -90, "")
ax3 = fig.add_subplot(133, projection="3d")
draw(ax3, result, (0.55, 0.75, 0.95), 0.55, 4, 0, "Side — cable exits rear (right)")
draw(ax3, ghost, (1.0, 0.55, 0.15), 0.75, 4, 0, "")
plt.tight_layout()
out_png = "/home/hatch/workspace/goals/mobile-pi-monitoring-rover/files/tfluna_holder_preview.png"
plt.savefig(out_png, dpi=150)
print("wrote", out_png)

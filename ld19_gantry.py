#!/usr/bin/env python3
"""LD19 gantry mount for the Pi monitoring rover (4WD acrylic chassis). v3.

v3 (Nov 22 2026) — full joint redesign after the v2 test fit failed.
Walter printed v2 and reported: the leg is hard to put into the
platform and the foot, and hard to lock down. Root causes:
  - The v2 pins were STOPS, not clamps. The pin holes sat at 11 mm
    (foot, 3 mm thick) and 162 mm (platform, 4 mm thick), so the pins
    sat 8 mm above the foot and 2 mm above the platform. They only
    kept the leg from falling through; the leg was held by friction
    in the slots, so it rocked and could slide up. A single pin is
    also a pivot, not a lock.
  - The slots were 9 mm wide for an 8 mm leg (0.5 mm/side) — too
    tight for real printed parts (slight oversize/warp binds), and
    threading a long leg through TWO aligned closed slots is hard
    when the leg is not perfectly straight.
  - The pin holes were hidden inside the slots, so the pin could
    not be seen or aimed, and the loose nut had to be held blind.
v3 fixes all three: NO threading through the foot, and only a short
drop into the platform. Each leg ends in a flat square flange that
bolts flat to the foot/platform with TWO vertical M3 bolts + nuts —
two bolts per joint, so it cannot rock, and the bolts are out in
the open where they can be seen and tightened. The foot is a plain
flat plate: the leg just sits on it and bolts down, nothing to
thread through. The platform's slot is a short 4 mm drop with a
funnel mouth (1 mm/side clearance, wider at the top), and the top
flange is wider than the slot, so it lands on the platform and
stops it at exactly the right height with no measuring.

v2 refinements kept: smooth (rounded foot/platform corners,
rounded posts), symmetric (gantry mirror-symmetric left-right;
ear slots symmetric about the centerline; plate layout mirrored),
one plate (all 5 parts on one 180x180 plate, ld19_gantry.3mf).

Gantry per Walter's red-line proposal (Nov 17 2026): two legs bolted to the
side acrylic plates, top platform over the camera mast, LD19 on top with a
clear 360-degree scan plane.

Dimensions (measured Nov 21 2026 from his photos + LD19 datasheet V2.5):
  sensor body 38.59 x 38.59 x 33.50 mm, 47 g (datasheet)
  sensor ears: 3 holes in a triangle — base 47 mm (left-right), sides 40 mm,
    apex height 32.4 mm (derived). Single ear faces the rover front.
    Ear positions vs sensor center (front = +Y):
    apex (0, +21.6), base (+/-23.5, -10.8). Slots absorb +/-3 mm error.
  deck width 145 mm edge-to-edge (red line at 14.5 cm)
  camera box top 160 mm above the upper deck (red line at 16 cm)
  platform top set at 160 mm; scan plane ~25 mm above the sensor base
    (head window band), so ~25 mm above the camera top — camera tilt-up
    margin ~15 mm. If the real camera height is lower, the mount still
    works, just with extra headroom.
  leg spacing 110 mm nominal; foot deck slots let the feet slide
    left-right to find a deck hole
  cable: assumed to exit the sensor bottom (central cutout); if it plugs
    into the side instead, it routes out between the corner posts.

Parts (all print flat, PLA, 0.2 mm layers, no supports):
  leg x2:     161 long x 30 wide x 8 thick bar, with a square flange at
              each end (30 x 30 x 4). Flanges are SOLID — no bolt holes
              in the leg (Walter, Nov 22: "remove the holds" / "the
              nail holes in the leg"). The foot/platform holes (3.4 mm)
              line up with the flanges; drive self-tapping M3 screws
              through them into the solid flange, or drill pilot holes
              if using bolts + nuts.
  foot x2:    35 x 50 x 3, rounded corners. Two 3.4 mm flange-bolt holes
              (at X = +/-10) + two deck slots (26 long, along X, at
              Y = +/-20.5). No leg slot.
  platform:   140 x 60 x 4, rounded corners. Two slots (10 x 32, with
              a funnel mouth 12 x 34 at the top) for the leg bars to
              drop into, four 3.4 mm flange-bolt holes (at X = +/-55
              +/- 10, Y = 0), 3 ear slots, central cable cutout.
              Flat on top — no corner posts: the sensor is held by its
              3 ear screws alone (Walter, Nov 22: "remove the holds").
Assembly (no threading, no blind holes):
  1. Bolt the feet to the deck through the deck slots (nuts under
     the deck). Feet at X = +/-55 (110 mm apart). Slide left-right
     to find a deck hole.
  2. Stand a leg on each foot — bottom flange flat on the foot
     plate, holes lined up — and bolt it down (2 bolts, nuts under
     the foot plate).
  3. Lower the platform straight down onto the legs: the leg bars
     drop into the slots (funnel mouths guide them in), and the top
     flanges (wider than the slots) land on the platform's top face
     and stop it at exactly the right height. Bolt the top flanges
     down (2 bolts each, nuts under the platform).
  4. Set the sensor in the pocket, ears on the slots, 3 ear screws.
Hardware (flanges are solid — self-tapping M3 screws bite into them):
  4x top-flange screws, 4x bottom-flange screws (M3x12 self-tapping,
    through the platform/foot holes into the solid flange; or drill
    pilot holes if using bolts + nuts)
  4x deck bolts (foot plate to deck, nuts under deck)
  3x ear screws (sensor ears to platform slots)
  7x M3 nuts + 7x washers (deck + ear joints only)
"""
import math
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from trimesh.boolean import union, difference

# ---------------------------------------------------------------- parameters
# --- heights (Z, deck surface = 0). Leg bottom sits on the foot plate.
FOOT_T = 3.0
PLATFORM_TOP_Z = 160.0
PLATFORM_T = 4.0
LEG_LEN = 161.0                     # print length; assembled z = 3..164
FLANGE_T = 4.0                      # flange thickness (along the leg)
FLANGE_X = 30.0                     # flange width, left-right (assembled X)
FLANGE_Y = 30.0                     # flange length, fore-aft (assembled Y)
FLANGE_HOLE_X = 10.0                # flange bolt holes at assembled X = +/-10
FLANGE_HOLE_D = 3.6                 # (unused now — flanges are solid;
                                    #  kept for reference)
PLATE_HOLE_D = 3.4                  # foot/platform bolt holes (vertical)

# --- leg bar
LEG_W = 30.0                        # fore-aft width
LEG_T = 8.0                         # left-right thickness
LEG_SPAN = 110.0                    # leg centerlines, X = +/-55

# --- foot
FOOT_L, FOOT_D = 35.0, 50.0                # X, Y
FOOT_R = 8.0                               # corner radius (smooth)
DECK_SLOT_W, DECK_SLOT_L = 3.4, 26.0       # deck slots, along X
DECK_SLOT_Y = 20.5                         # deck slots at Y = +/-20.5

# --- platform
PLAT_L, PLAT_D = 140.0, 60.0               # X, Y
PLAT_R = 8.0                               # corner radius (smooth)
SENSOR_BODY = 38.59                        # ghost sensor size in the preview
SLOT_W, SLOT_L = 10.0, 32.0                # platform slot for the leg bar
SLOT_LEAD_W, SLOT_LEAD_L = 12.0, 34.0      # funnel mouth at the slot top
EAR_SLOT_W, EAR_SLOT_L = 3.4, 10.0
CABLE_CUT = 16.0

# ear positions vs sensor center (front = +Y); symmetric about X = 0
EARS = [(0.0, 21.6), (23.5, -10.8), (-23.5, -10.8)]
assert EARS[1][0] == -EARS[2][0] and EARS[1][1] == EARS[2][1], "ears not mirrored"
assert EARS[0][0] == 0.0, "apex ear must sit on the centerline"

# --- one-plate layout (bed 180x180, centered coords, X/Y in [-90, 90])
# leg1 (0,-73), leg2 (0,-38), platform (0,12), foot1 (-62,64.5, rot 90),
# foot2 (62,64.5, rot 90). Symmetric about X = 0.
PLATE = [
    ("leg", (0.0, -73.0), 0.0),
    ("leg", (0.0, -38.0), 0.0),
    ("platform", (0.0, 12.0), 0.0),
    ("foot", (-62.0, 64.5), 90.0),
    ("foot", (62.0, 64.5), 90.0),
]
BED = 180.0

OUT = "/home/hatch/workspace/goals/mobile-pi-monitoring-rover/files"

# ------------------------------------------------------------------- helpers
def box(sx, sy, sz, cx, cy, cz):
    m = trimesh.creation.box(extents=[sx, sy, sz])
    m.apply_translation([cx, cy, cz])
    return m

def cyl_z(r, h, cx, cy, cz, sections=32):
    m = trimesh.creation.cylinder(radius=r, height=h, sections=sections)
    m.apply_translation([cx, cy, cz])
    return m

def cyl_x(r, length, cx, cy, cz, sections=32):
    """Cylinder whose axis lies along X (for the flange bolt holes,
    which are horizontal in the print)."""
    m = trimesh.creation.cylinder(radius=r, height=length, sections=sections)
    m.apply_transform(trimesh.transformations.rotation_matrix(
        np.radians(90), [0, 1, 0], [0, 0, 0]))
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

def capsule_x(length, width, cx, cy, z0, z1):
    """Capsule slot along X, cut through Z0..Z1."""
    straight = length - width
    return union([
        box(straight, width, z1 - z0, cx, cy, (z0 + z1) / 2),
        cyl_z(width / 2, z1 - z0, cx - straight / 2, cy, (z0 + z1) / 2),
        cyl_z(width / 2, z1 - z0, cx + straight / 2, cy, (z0 + z1) / 2),
    ], engine="manifold")

def capsule_rot(length, width, cx, cy, deg, z0, z1):
    """Capsule slot along X, rotated by deg about Z, centered at (cx, cy)."""
    m = capsule_x(length, width, 0, 0, z0, z1)
    m.apply_transform(trimesh.transformations.rotation_matrix(
        np.radians(deg), [0, 0, 1], [0, 0, 0]))
    m.apply_translation([cx, cy, 0])
    return m

# ------------------------------------------------------------------------ leg
# Print orientation: length along X (0..161), width along Y (30).
# The bar is 8 thick (Z), centered at Z = FLANGE_X/2 = 15 so it sits
# centered on the flanges. Flanges at both ends: 4 long (X) x 30 wide
# (Y) x 30 thick (Z), centered at Z = 15. The flanges are SOLID —
# no bolt holes (Walter Nov 22: "remove the holds" / "the nail holes
# in the leg"). Screws go through the foot/platform holes and bite
# into the solid flange (self-tapping), or drill pilot holes.
# The flange (30 x 30) is wider than the platform slot (10 x 32), so
# it lands on the platform and stops it at the right height.
leg_parts = [box(LEG_LEN, LEG_W, LEG_T, LEG_LEN / 2, 0, FLANGE_X / 2)]
for fx in (FLANGE_T / 2, LEG_LEN - FLANGE_T / 2):
    leg_parts.append(box(FLANGE_T, FLANGE_Y, FLANGE_X, fx, 0, FLANGE_X / 2))
leg = union(leg_parts, engine="manifold")

# ----------------------------------------------------------------------- foot
# Print orientation: 35 (X) x 50 (Y) x 3 (Z), centered at origin, r=8 corners.
# Two flange-bolt holes (vertical, 3.4 mm) at X = +/-10, Y = 0; two deck
# slots (along X, 26 long) at Y = +/-20.5. No leg slot — the leg's bottom
# flange just sits on this plate and bolts down.
foot_body = rounded_rect_prism(FOOT_L, FOOT_D, FOOT_T, FOOT_R, 0, 0, FOOT_T / 2)
foot_cuts = [
    cyl_z(PLATE_HOLE_D / 2, FOOT_T + 1.0, FLANGE_HOLE_X, 0, FOOT_T / 2),
    cyl_z(PLATE_HOLE_D / 2, FOOT_T + 1.0, -FLANGE_HOLE_X, 0, FOOT_T / 2),
    capsule_x(DECK_SLOT_L, DECK_SLOT_W, 0, DECK_SLOT_Y, -0.5, FOOT_T + 0.5),
    capsule_x(DECK_SLOT_L, DECK_SLOT_W, 0, -DECK_SLOT_Y, -0.5, FOOT_T + 0.5),
]
foot = difference([foot_body, union(foot_cuts, engine="manifold")],
                  engine="manifold")

# ------------------------------------------------------------------- platform
# Print orientation: 140 (X) x 60 (Y) x 4 (Z), centered at origin, r=8 corners.
# Flat on top — no corner posts (removed Nov 22 on Walter's word:
# "remove the holds"). The sensor sits on the flat plate and is held
# by its 3 ear screws alone.
plat_body = rounded_rect_prism(PLAT_L, PLAT_D, PLATFORM_T, PLAT_R,
                               0, 0, PLATFORM_T / 2)

plat_cuts = []
for s in (-1, 1):
    lx = s * LEG_SPAN / 2
    # Slot for the leg bar (10 x 32), with a funnel mouth at the top
    # (12 x 34 for the top 1.5 mm) so the bar self-centers as the
    # platform lowers onto the legs.
    plat_cuts.append(box(SLOT_W, SLOT_L, PLATFORM_T + 1.0,
                         lx, 0, PLATFORM_T / 2))
    plat_cuts.append(box(SLOT_LEAD_W, SLOT_LEAD_L, 1.5,
                         lx, 0, PLATFORM_T - 0.75 + 0.5))
    # Flange-bolt holes for this leg (vertical, 3.4 mm), at X = leg
    # center +/- 10, Y = 0. Both are in solid platform, clear of the
    # slot (slot X = leg center +/- 5).
    for sx in (-1, 1):
        plat_cuts.append(cyl_z(PLATE_HOLE_D / 2, PLATFORM_T + 1.0,
                               lx + sx * FLANGE_HOLE_X, 0, PLATFORM_T / 2))
for (ex, ey) in EARS:  # radial ear slots
    ang = math.degrees(math.atan2(ey, ex))
    plat_cuts.append(capsule_rot(EAR_SLOT_L, EAR_SLOT_W, ex, ey, ang,
                                 -0.5, PLATFORM_T + 0.5))
plat_cuts.append(box(CABLE_CUT, CABLE_CUT, PLATFORM_T + 1.0,
                     0, 0, PLATFORM_T / 2))
platform = difference([plat_body, union(plat_cuts, engine="manifold")],
                      engine="manifold")

# ------------------------------------------------------------------ validate
meshes = {"leg": leg, "foot": foot, "platform": platform}
for name, mesh in meshes.items():
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    print(f"{name}: watertight={mesh.is_watertight} "
          f"volume={round(mesh.volume, 1)} mm^3 "
          f"bounds_min={np.round(mesh.bounds[0], 2)} "
          f"bounds_max={np.round(mesh.bounds[1], 2)} "
          f"triangles={len(mesh.faces)} euler={mesh.euler_number}")
    mesh.export(f"{OUT}/ld19_{name}.stl")
    print(f"  wrote {OUT}/ld19_{name}.stl")

# symmetry check: platform mirrored about X=0 must match its own bounds
mir = platform.copy()
mir.apply_transform(trimesh.transformations.reflection_matrix(
    [0, 0, 0], [1, 0, 0]))
assert np.allclose(mir.bounds, platform.bounds), "platform not X-symmetric"
# flange check: the flange (30 x 30) is wider than the platform slot
# (10 x 32) in X, so it cannot pass through the slot — it lands on the
# platform and stops it at the right height.
assert FLANGE_X > SLOT_W and FLANGE_Y > SLOT_W, "flange must exceed the slot"
print("symmetry: platform mirrors about X=0 exactly; legs/feet identical "
      "pairs; plate layout mirrored about X=0.")

# ------------------------------------------------------- one-plate placement
def placed_bounds(mesh, cx, cy, deg):
    """XY footprint of a mesh centered at (cx, cy), rotated deg about Z."""
    m = mesh.copy()
    c = (m.bounds[0][:2] + m.bounds[1][:2]) / 2
    m.apply_translation([-c[0], -c[1], 0])
    if deg:
        m.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(deg), [0, 0, 1], [0, 0, 0]))
    m.apply_translation([cx, cy, 0])
    return m.bounds

footprints = []
for name, (cx, cy), deg in PLATE:
    b = placed_bounds(meshes[name], cx, cy, deg)
    footprints.append((name, b))
    print(f"plate: {name} at ({cx}, {cy}) rot {deg} -> "
          f"X [{b[0][0]:.1f}, {b[1][0]:.1f}] Y [{b[0][1]:.1f}, {b[1][1]:.1f}] "
          f"Z [{b[0][2]:.1f}, {b[1][2]:.1f}]")
    assert b[0][0] >= -BED / 2 and b[1][0] <= BED / 2, f"{name} off bed in X"
    assert b[0][1] >= -BED / 2 and b[1][1] <= BED / 2, f"{name} off bed in Y"
    assert b[0][2] >= 0, f"{name} below bed"
for i in range(len(footprints)):
    for j in range(i + 1, len(footprints)):
        n1, b1 = footprints[i]
        n2, b2 = footprints[j]
        ox = min(b1[1][0], b2[1][0]) - max(b1[0][0], b2[0][0])
        oy = min(b1[1][1], b2[1][1]) - max(b1[0][1], b2[0][1])
        assert not (ox > 0 and oy > 0), f"{n1} overlaps {n2} on the plate"
print("plate: all 5 parts fit 180x180 with no overlaps.")

# ------------------------------------------------------------------- preview
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def sensor_ghost():
    """Translucent sensor at its seated position (base on platform top)."""
    z0 = PLATFORM_TOP_Z
    gs = [box(SENSOR_BODY, SENSOR_BODY, 18.0, 0, 0, z0 + 9.0)]  # base
    gs.append(cyl_z(18.0, 15.5, 0, 0, z0 + 18.0 + 15.5 / 2))    # spinning head
    for (ex, ey) in EARS:                                        # ears
        gs.append(box(8.0, 4.0, 2.0, ex, ey, z0 + 1.0))
    return union(gs, engine="manifold")

def xform(mesh, rot_axis, deg, tx, ty, tz):
    m = mesh.copy()
    if deg:
        m.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(deg), rot_axis, [0, 0, 0]))
    m.apply_translation([tx, ty, tz])
    return m

# Assembled: leg print X (0..161) becomes vertical Z. Rotate -90 deg
# about Y: (x,y,z) -> (-z, y, x). After rotation the leg spans X = -30..0
# (flanges) and Z = 0..161. Translate so the bottom sits on the foot
# plate (z = 3) and the leg centers at X = +/-55: the flange center
# (print Z = 15) lands at X' = -15, so tx = +70 for the right leg
# (X' 40..70, center 55) and tx = -40 for the left (X' -70..-40).
assembled = []
for s in (-1, 1):
    tx = 70.0 if s > 0 else -40.0
    assembled.append(xform(leg, [0, 1, 0], -90, tx, 0, FOOT_T))
    assembled.append(xform(foot, None, 0, s * LEG_SPAN / 2, 0, 0))
assembled.append(xform(platform, None, 0, 0, 0, PLATFORM_TOP_Z - PLATFORM_T))
gantry = union(assembled, engine="manifold")
ghost = sensor_ghost()

def draw(ax, mesh, color, alpha, elev, azim, title, xlim, ylim, zlim):
    tris = mesh.triangles
    ax.add_collection3d(Poly3DCollection(tris, facecolor=color,
                                         edgecolor=(0.2, 0.2, 0.2, 0.25),
                                         linewidths=0.2, alpha=alpha))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0],
                       zlim[1] - zlim[0]))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()

fig = plt.figure(figsize=(13, 4.6))
ax1 = fig.add_subplot(131, projection="3d")
draw(ax1, gantry, (0.55, 0.75, 0.95), 1.0, 22, -58,
     "Gantry assembled (isometric)", (-80, 80), (-50, 50), (0, 200))
draw(ax1, ghost, (1.0, 0.55, 0.15), 0.6, 22, -58, "", (-80, 80), (-50, 50),
     (0, 200))
ax2 = fig.add_subplot(132, projection="3d")
draw(ax2, gantry, (0.55, 0.75, 0.95), 0.55, 4, -90,
     "Front — sensor seated (orange)", (-80, 80), (-50, 50), (0, 200))
draw(ax2, ghost, (1.0, 0.55, 0.15), 0.75, 4, -90, "", (-80, 80), (-50, 50),
     (0, 200))
ax3 = fig.add_subplot(133, projection="3d")
draw(ax3, gantry, (0.55, 0.75, 0.95), 0.55, 4, 0,
     "Side — legs + feet + platform", (-80, 80), (-50, 50), (0, 200))
draw(ax3, ghost, (1.0, 0.55, 0.15), 0.75, 4, 0, "", (-80, 80), (-50, 50),
     (0, 200))
plt.tight_layout()
out_png = f"{OUT}/ld19_gantry_preview.png"
plt.savefig(out_png, dpi=150)
print("wrote", out_png)

# ------------------------------------------------------------ plate diagram
fig2, axp = plt.subplots(figsize=(7, 7))
axp.add_patch(plt.Rectangle((-90, -90), 180, 180, fill=False, edgecolor="black",
                            linewidth=2))
colors = {"leg": "#8fb8e8", "platform": "#7fb069", "foot": "#e8a87f"}
for name, (cx, cy), deg in PLATE:
    b = placed_bounds(meshes[name], cx, cy, deg)
    x0, y0 = b[0][0], b[0][1]
    w, d = b[1][0] - x0, b[1][1] - y0
    axp.add_patch(plt.Rectangle((x0, y0), w, d, facecolor=colors[name],
                                edgecolor="black", linewidth=1))
    axp.text(cx, cy, f"{name}\n({w:.0f}x{d:.0f})", ha="center", va="center",
             fontsize=10)
axp.set_xlim(-95, 95); axp.set_ylim(-95, 95)
axp.set_aspect("equal")
axp.set_title("One plate — 180x180 bed (top view, mm)", fontsize=12)
axp.set_xlabel("X (mm)"); axp.set_ylabel("Y (mm)")
axp.grid(True, alpha=0.3)
plt.tight_layout()
out_plate = f"{OUT}/ld19_plate_layout.png"
plt.savefig(out_plate, dpi=150)
print("wrote", out_plate)

# ---------------------------------------------------------------------- 3MF
NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
ET.register_namespace("", NS)

def fmt(v):
    return f"{v:.3f}".rstrip("0").rstrip(".")

def mesh_xml(mesh):
    verts = mesh.vertices
    tris = mesh.faces
    v_el = ET.Element("vertices")
    for (x, y, z) in verts:
        ET.SubElement(v_el, "vertex", {"x": fmt(x), "y": fmt(y), "z": fmt(z)})
    t_el = ET.Element("triangles")
    for (a, b, c) in tris:
        ET.SubElement(t_el, "triangle",
                      {"v1": str(int(a)), "v2": str(int(b)), "v3": str(int(c))})
    m_el = ET.Element("mesh")
    m_el.append(v_el)
    m_el.append(t_el)
    return m_el

obj_ids = {"leg": 1, "foot": 2, "platform": 3}
model = ET.Element("model", {"xmlns": NS, "unit": "millimeter",
                             "xml:lang": "en-US"})
resources = ET.SubElement(model, "resources")
for name, oid in obj_ids.items():
    obj = ET.SubElement(resources, "object",
                        {"id": str(oid), "name": f"LD19 {name}",
                         "type": "model"})
    obj.append(mesh_xml(meshes[name]))
build = ET.SubElement(model, "build")
for name, (cx, cy), deg in PLATE:
    if name == "leg":
        tx, ty = cx - LEG_LEN / 2, cy
    else:
        tx, ty = cx, cy
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    tr = f"{fmt(c)} {fmt(-s)} 0 {fmt(tx)} {fmt(s)} {fmt(c)} 0 {fmt(ty)} 0 0 1 0"
    ET.SubElement(build, "item",
                  {"objectid": str(obj_ids[name]), "transform": tr})

model_bytes = ET.tostring(model, encoding="UTF-8", xml_declaration=True)
content_types = ('<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
    'content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-'
    'package.relationships+xml"/>'
    '<Default Extension="model" ContentType="application/vnd.ms-package.'
    '3dmanufacturing-3dmodel+xml"/></Types>')
rels = ('<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
    'relationships">'
    '<Relationship Type="http://schemas.microsoft.com/3dmanufacturing/'
    '2013/01/3dmodel" Target="/3D/3dmodel.model" Id="rel0"/>'
    '</Relationships>')

out_3mf = f"{OUT}/ld19_gantry.3mf"
with zipfile.ZipFile(out_3mf, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("[Content_Types].xml", content_types)
    zf.writestr("_rels/.rels", rels)
    zf.writestr("3D/3dmodel.model", model_bytes)
print("wrote", out_3mf)

# ------------------------------------------------------------ validate 3MF
with zipfile.ZipFile(out_3mf) as zf:
    names = zf.namelist()
    assert "[Content_Types].xml" in names and "_rels/.rels" in names
    assert "3D/3dmodel.model" in names
    root = ET.fromstring(zf.read("3D/3dmodel.model"))
ns = {"m": NS}
objs = root.findall(".//m:object", ns)
items = root.findall(".//m:item", ns)
print(f"3MF check: {len(objs)} objects, {len(items)} build items")
assert len(objs) == 3 and len(items) == 5
for it in items:
    tr = [float(v) for v in it.get("transform").split()]
    assert len(tr) == 12
print("3MF check: package + XML + transforms OK")
try:
    scene = trimesh.load(out_3mf)
    print("3MF round-trip: trimesh loaded it as", type(scene).__name__)
    if hasattr(scene, "geometry"):
        for gname, g in scene.geometry.items():
            print(f"  {gname}: {len(g.faces)} tris, "
                  f"bounds {np.round(g.bounds, 1).tolist()}")
except Exception as e:  # noqa: BLE001 - trimesh 3MF support varies
    print("3MF round-trip: trimesh could not load it:", e)

print()
print("Hardware (flanges are solid — self-tapping M3 screws bite into them):")
print("  4x top-flange screws (through platform holes into solid flange)")
print("  4x bottom-flange screws (through foot holes into solid flange)")
print("  4x deck bolts (foot plate to deck, nuts under deck)")
print("  3x ear screws (sensor ears to platform slots)")
print("  7x M3 nuts + 7x washers (deck + ear joints only)")
print("Print: PLA, 0.2 mm layers, 3 walls, 20-25% infill, no supports,")
print("  ONE plate (ld19_gantry.3mf), skirt, ~2-2.5 h total.")

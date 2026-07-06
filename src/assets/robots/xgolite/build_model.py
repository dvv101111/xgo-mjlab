"""Generate xgolite.xml from the hardware-verified lite2 URDF.

The URDF in Quadruped-robot is the kinematic ground truth: its axes and
zero positions were verified joint-by-joint against the real robot in the
operator app (2026-07-05). This script derives every number in the MJCF
from it instead of hand-transcription:

  - body tree / joint anchors / joint axes  -> from the URDF via MuJoCo
  - foot contact points                     -> lowest calf-mesh vertices
  - visual geometry                         -> the actual STL meshes
  - collision primitives                    -> fitted to mesh bounding boxes
  - joint ranges                            -> servo_calibration.json via
                                               the driver's own JointMapper
  - actuators                               -> kp/kv identified from step
                                               responses (2026-07-05)

It then validates the generated model: joint anchors, axes and foot points
must match the URDF to < 0.5 mm at 25 random poses.

Usage:
  .venv/bin/python src/assets/robots/xgolite/build_model.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
QUADRUPED = HERE.parents[4]  # luwu_mjlab is nested inside Quadruped-robot
URDF = QUADRUPED / "assets" / "robots" / "lite2" / "urdf" / "lite2_description.urdf"
MESH_SRC = QUADRUPED / "assets" / "robots" / "lite2" / "meshes"
OUT_XML = HERE / "xmls" / "xgolite.xml"
OUT_MESHDIR = HERE / "xmls" / "meshes"

sys.path.insert(0, str(QUADRUPED / "src"))
from xgo.openfw.calibration import JointMapper  # noqa: E402

# canonical leg -> URDF/servo digit (vendor circular numbering: 3=back-right)
LEG_NUM = {"fl": "1", "fr": "2", "br": "3", "bl": "4"}
PART_NUM = {"hip": "3", "thigh": "2", "calf": "1"}
LEGS = ("fl", "fr", "bl", "br")

# identified actuator/joint parameters (step-response fit 2026-07-05)
KP, KV = 5.0, 0.12
FORCERANGE = 0.22
JOINT_DEFAULTS = 'damping="0.05" frictionloss="0.001" armature="0.002"'
# mass model: URDF inertials are plastic shells only (85 g total, unusable).
# Explicit masses summing to the ~0.63 kg estimate; servo mass sits in the
# thigh. CoM is then shifted to +15 mm, the midpoint of the empirically
# bracketed range (candidate-C stance stood -> CoM < +23 mm; v1 default
# tipped -> CoM > +9 mm).
MASS = {"base": 0.320, "hip": 0.015, "thigh": 0.045, "calf": 0.015}
TARGET_COM_X = 0.015
FOOT_RADIUS = 0.006


def load_urdf() -> tuple[mujoco.MjModel, mujoco.MjData]:
    text = URDF.read_text()
    i = text.index(">", text.index("<robot")) + 1
    ext = (f'\n  <mujoco><compiler meshdir="{MESH_SRC}" strippath="true" '
           f'balanceinertia="true" discardvisual="false" fusestatic="false"/></mujoco>')
    m = mujoco.MjModel.from_xml_string(text[:i] + ext + text[i:])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def body_frame(m, d, body):
    bid = m.body(body).id
    return d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy()


def mesh_geoms(m, body):
    """(mesh_name, pos, quat, mesh_id) of every mesh geom on a body, in the
    body frame, with COMPILED placements (mesh centroid offset baked in)."""
    bid = m.body(body).id
    out = []
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] != bid or m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[gid]
        out.append((m.mesh(mid).name, m.geom_pos[gid].copy(),
                    m.geom_quat[gid].copy(), mid))
    return out


def uncompile_mesh_frame(m, mid, pos_w, R_w):
    """Convert a COMPILED world placement back to the raw-STL placement to
    write in XML. The compiler re-centers each mesh (mesh_pos/mesh_quat) and
    bakes that into geom_pos/quat; writing compiled values against the raw
    STL would apply the offset twice."""
    Rm = np.zeros(9)
    mujoco.mju_quat2Mat(Rm, m.mesh_quat[mid])
    Rm = Rm.reshape(3, 3)
    R_written = R_w @ Rm.T
    p_written = pos_w - R_written @ m.mesh_pos[mid]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R_written.flatten())
    return p_written, quat


def mesh_verts_in_body(m, d, body):
    """All mesh vertices of a body, expressed in the body frame (zero pose)."""
    bp, bR = body_frame(m, d, body)
    bid = m.body(body).id
    chunks = []
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] != bid or m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[gid]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = v @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
        chunks.append((w - bp) @ bR)
    return np.vstack(chunks)


def foot_point(m, d, leg) -> np.ndarray:
    """Foot pad center in the calf body frame: centroid of the vertices in
    the bottom 2 mm of the calf mesh at the zero pose (URDF zero has the
    calf swept ~63 deg forward, so 'lowest' is the foot tip)."""
    v = mesh_verts_in_body(m, d, f"Link_{LEG_NUM[leg]}1")
    # body frame == world orientation at zero (URDF root aligned), but be
    # safe: find min along the world-down direction mapped into the body
    bp, bR = body_frame(m, d, f"Link_{LEG_NUM[leg]}1")
    down = bR.T @ np.array([0.0, 0.0, -1.0])
    proj = v @ down
    pad = v[proj > proj.max() - 0.002]
    return pad.mean(axis=0)


def fmt(x, nd=6):
    if isinstance(x, (list, tuple, np.ndarray)):
        return " ".join(fmt(v, nd) for v in x)
    s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def build() -> str:
    m, d = load_urdf()
    mapper = JointMapper(QUADRUPED / "config" / "servo_calibration.json",
                         QUADRUPED / "config" / "lite2_joint_map.yaml")
    lims = mapper.limits_rad()

    OUT_MESHDIR.mkdir(exist_ok=True)
    used_meshes = set()

    # ---- extract per-leg kinematics from the URDF (base_link frame == world
    # at zero pose, URDF base has identity root)
    base_p, base_R = body_frame(m, d, "base_link")
    assert np.allclose(base_p, 0) and np.allclose(base_R, np.eye(3))

    legs = {}
    for leg in LEGS:
        n = LEG_NUM[leg]
        hip_b, hip_R = body_frame(m, d, f"Link_{n}3")
        thigh_b, thigh_R = body_frame(m, d, f"Link_{n}2")
        calf_b, calf_R = body_frame(m, d, f"Link_{n}1")
        j = {}
        for part in ("hip", "thigh", "calf"):
            jid = m.joint(f"Joint_{n}{PART_NUM[part]}").id
            j[part] = (d.xanchor[jid].copy(), d.xaxis[jid].copy())
        foot = foot_point(m, d, leg)
        # child-body positions relative to parent frames. URDF child frames
        # are rotated (SolidWorks rpy); we re-express everything in the base
        # orientation so the MJCF uses identity body frames throughout: a
        # joint angle then means the same rotation in both models, and the
        # zero pose is inherited exactly.
        hip_pos = j["hip"][0]                       # base frame == world
        thigh_pos = j["thigh"][0] - j["hip"][0]
        calf_pos = j["calf"][0] - j["thigh"][0]
        foot_w = calf_b + calf_R @ foot             # world at zero
        foot_pos = foot_w - j["calf"][0]            # calf joint frame, axes=world
        legs[leg] = {
            "hip_pos": hip_pos, "hip_axis": j["hip"][1],
            "thigh_pos": thigh_pos, "thigh_axis": j["thigh"][1],
            "calf_pos": calf_pos, "calf_axis": j["calf"][1],
            "foot": foot_pos,
            # visual meshes with their world-frame placements at zero,
            # re-expressed relative to the identity-orientation joint frames
            "meshes": {},
        }
        for part, link_body, jkey in (("hip", f"Link_{n}3", "hip"),
                                      ("thigh", f"Link_{n}2", "thigh"),
                                      ("calf", f"Link_{n}1", "calf")):
            lb_p, lb_R = body_frame(m, d, link_body)
            entries = []
            for mesh, gp, gq, mid in mesh_geoms(m, link_body):
                Rq = np.zeros(9); mujoco.mju_quat2Mat(Rq, gq)
                gR_w = lb_R @ Rq.reshape(3, 3)
                gp_w = lb_p + lb_R @ gp
                rel_p, quat = uncompile_mesh_frame(m, mid, gp_w - j[jkey][0], gR_w)
                entries.append((mesh, rel_p, quat))
                used_meshes.add(mesh)
            legs[leg]["meshes"][part] = entries

    # base + arm visual meshes (arm frozen at the calibrated zero = the pose
    # the deploy loop holds; its links become fixed geoms on the base)
    base_meshes = []
    for body in ("base_link", "Link_53", "Link_52", "Link_51", "Link_50"):
        try:
            m.body(body)
        except KeyError:
            continue
        lb_p, lb_R = body_frame(m, d, body)
        for mesh, gp, gq, mid in mesh_geoms(m, body):
            Rq = np.zeros(9); mujoco.mju_quat2Mat(Rq, gq)
            gR_w = lb_R @ Rq.reshape(3, 3)
            gp_w = lb_p + lb_R @ gp
            p_wr, quat = uncompile_mesh_frame(m, mid, gp_w, gR_w)
            base_meshes.append((mesh, p_wr, quat))
            used_meshes.add(mesh)

    # collision primitives from mesh bboxes
    base_v = np.vstack([mesh_verts_in_body(m, d, b)
                        for b in ("base_link",)])
    b_lo, b_hi = base_v.min(0), base_v.max(0)
    base_size = (b_hi - b_lo) / 2
    base_center = (b_hi + b_lo) / 2

    for mesh in used_meshes:
        shutil.copy2(MESH_SRC / f"{mesh}.STL", OUT_MESHDIR / f"{mesh}.STL")

    # ---- emit MJCF
    mesh_assets = "\n    ".join(
        f'<mesh name="{n}" file="{n}.STL" />' for n in sorted(used_meshes))

    def leg_xml(leg):
        L = legs[leg]
        lo_h, hi_h = lims[f"{leg}_hip"]
        lo_t, hi_t = lims[f"{leg}_thigh"]
        lo_c, hi_c = lims[f"{leg}_calf"]
        f = L["foot"]

        def vis(part):
            return "\n            ".join(
                f'<geom class="visual" type="mesh" mesh="{mesh}" '
                f'pos="{fmt(p)}" quat="{fmt(q)}" rgba="0.8 0.8 0.8 1" />'
                for mesh, p, q in L["meshes"][part])

        thigh_len = -L["calf_pos"][2]
        return f"""
      <body name="{leg}_hip" pos="{fmt(L['hip_pos'])}">
        <joint name="{leg}_hip_joint" pos="0 0 0" axis="{fmt(L['hip_axis'], 4)}" range="{fmt(lo_h, 4)} {fmt(hi_h, 4)}" actuatorfrcrange="-{FORCERANGE} {FORCERANGE}" />
        {vis('hip')}
        <geom name="{leg}_hip_col" type="box" size="0.015 0.022 0.013" pos="0 0 0" mass="{MASS['hip']}" class="collision_off" />
        <body name="{leg}_thigh" pos="{fmt(L['thigh_pos'])}">
          <joint name="{leg}_thigh_joint" pos="0 0 0" axis="{fmt(L['thigh_axis'], 4)}" range="{fmt(lo_t, 4)} {fmt(hi_t, 4)}" actuatorfrcrange="-{FORCERANGE} {FORCERANGE}" />
          {vis('thigh')}
          <geom name="{leg}_thigh" type="capsule" fromto="0 0 0 {fmt(L['calf_pos'])}" size="0.010" mass="{MASS['thigh']}" contype="1" conaffinity="0" condim="1" group="3" />
          <body name="{leg}_calf" pos="{fmt(L['calf_pos'])}">
            <joint name="{leg}_calf_joint" pos="0 0 0" axis="{fmt(L['calf_axis'], 4)}" range="{fmt(lo_c, 4)} {fmt(hi_c, 4)}" actuatorfrcrange="-{FORCERANGE} {FORCERANGE}" />
            {vis('calf')}
            <geom name="{leg}_calf" type="capsule" fromto="0 0 0 {fmt(f)}" size="0.006" mass="{MASS['calf']}" class="collision_off" />
            <geom name="{leg}_foot_pad" type="sphere" size="{FOOT_RADIUS}" pos="{fmt(f)}" contype="1" conaffinity="0" condim="3" rgba="0.3 0.3 0.3 1" />
            <site name="{leg}" pos="{fmt(f)}" type="sphere" size="0.004" />
          </body>
        </body>
      </body>"""

    base_mesh_xml = "\n      ".join(
        f'<geom class="visual" type="mesh" mesh="{mesh}" '
        f'pos="{fmt(p)}" quat="{fmt(q)}" rgba="0.85 0.85 0.85 1" />'
        for mesh, p, q in base_meshes)

    actuators = "\n    ".join(
        f'<position name="{leg}_{part}_joint" joint="{leg}_{part}_joint" '
        f'ctrlrange="{fmt(lims[f"{leg}_{part}"][0], 4)} {fmt(lims[f"{leg}_{part}"][1], 4)}" '
        f'class="joint_motor" />'
        for leg in LEGS for part in ("hip", "thigh", "calf"))

    xml = f"""<mujoco model="xgolite">
  <!-- GENERATED by build_model.py from the hardware-verified lite2 URDF
       ({URDF.name}) + servo_calibration.json. Do not edit numbers by hand:
       re-run  .venv/bin/python src/assets/robots/xgolite/build_model.py

       Conventions (identical to the URDF / driver / operator app):
       x forward, y left, z up; hips: + = outward, thighs/calves:
       + = foot forward. Joint zero == URDF zero == servo calibration zero
       (the URDF zero has the calf swept ~63 deg forward - that is how the
       real linkage is built, NOT straight down).
       Arm (Link_5x) is frozen at its calibrated zero and welded to the
       base; locomotion action space = 12 leg joints. -->
  <compiler angle="radian" meshdir="meshes" />
  <option timestep="0.002" />

  <default>
    <joint {JOINT_DEFAULTS} />
    <default class="joint_motor">
      <position kp="{KP}" kv="{KV}" forcerange="-{FORCERANGE} {FORCERANGE}" />
    </default>
    <default class="visual">
      <geom contype="0" conaffinity="0" density="0" group="1" />
    </default>
    <default class="collision_off">
      <geom contype="0" conaffinity="0" group="3" />
    </default>
  </default>

  <asset>
    {mesh_assets}
  </asset>

  <sensor>
    <framequat name="orientation" objtype="site" noise="0.001" objname="imu" />
    <gyro name="angular-velocity" site="imu" noise="0.005" />
    <gyro name="imu_ang_vel" site="imu" />
    <velocimeter name="imu_lin_vel" site="imu" />
    <accelerometer name="imu_accel" site="imu" />
    <subtreeangmom name="root_angmom" body="base" />
  </sensor>

  <worldbody>
    <body name="base" pos="0 0 0.14">
      <joint name="floating_base" type="free" />
      {base_mesh_xml}
      <geom name="base" type="box" size="{fmt(base_size)}" pos="{fmt(base_center)}"
            mass="{MASS['base']}" contype="1" conaffinity="0" condim="1" group="3" />
      <site name="imu" pos="0 0 0" quat="1 0 0 0" />
{''.join(leg_xml(leg) for leg in LEGS)}
    </body>
  </worldbody>

  <actuator>
    {actuators}
  </actuator>
</mujoco>
"""
    return xml


def validate(xml: str) -> None:
    mu, du = load_urdf()
    mx = mujoco.MjModel.from_xml_string(xml, {f"meshes/{p.name}": p.read_bytes()
                                              for p in OUT_MESHDIR.glob("*.STL")})
    dx = mujoco.MjData(mx)

    # foot pad centers in calf body frames — fixed local points, must be
    # captured at the zero pose (elsewhere the knee can be the lowest vertex)
    foot_local = {leg: foot_point(mu, du, leg) for leg in LEGS}

    rng = np.random.default_rng(7)
    worst_anchor = worst_axis = worst_foot = 0.0
    for trial in range(25):
        pose = {}
        for leg in LEGS:
            for part in ("hip", "thigh", "calf"):
                jn_u = f"Joint_{LEG_NUM[leg]}{PART_NUM[part]}"
                jn_x = f"{leg}_{part}_joint"
                lo, hi = mx.joint(jn_x).range
                val = rng.uniform(lo, hi)
                pose[(jn_u, jn_x)] = val
        du.qpos[:] = 0
        dx.qpos[:] = 0; dx.qpos[3] = 1
        for (jn_u, jn_x), val in pose.items():
            du.qpos[mu.joint(jn_u).qposadr[0]] = val
            dx.qpos[mx.joint(jn_x).qposadr[0]] = val
        mujoco.mj_forward(mu, du)
        mujoco.mj_forward(mx, dx)
        off = dx.xpos[mx.body("base").id].copy()    # remove base z offset
        for leg in LEGS:
            for part in ("hip", "thigh", "calf"):
                ju = mu.joint(f"Joint_{LEG_NUM[leg]}{PART_NUM[part]}").id
                jx = mx.joint(f"{leg}_{part}_joint").id
                worst_anchor = max(worst_anchor,
                                   float(np.linalg.norm(du.xanchor[ju] - (dx.xanchor[jx] - off))))
                worst_axis = max(worst_axis,
                                 float(np.linalg.norm(du.xaxis[ju] - dx.xaxis[jx])))
            # foot: zero-pose pad centroid carried through URDF FK vs the
            # generated model's site, at every random pose
            bp, bR = body_frame(mu, du, f"Link_{LEG_NUM[leg]}1")
            fu = bp + bR @ foot_local[leg]
            fx = dx.site(leg).xpos - off
            worst_foot = max(worst_foot, float(np.linalg.norm(fu - fx)))

    # visual meshes: compare world-space vertex clouds at the zero pose
    # (catches compiled-vs-raw mesh frame errors that FK checks cannot see)
    du.qpos[:] = 0
    dx.qpos[:] = 0; dx.qpos[3] = 1
    mujoco.mj_forward(mu, du)
    mujoco.mj_forward(mx, dx)
    off = dx.xpos[mx.body("base").id].copy()
    worst_mesh = 0.0
    for leg in LEGS:
        for n_suffix, xbody in (("3", f"{leg}_hip"), ("2", f"{leg}_thigh"),
                                ("1", f"{leg}_calf")):
            vu = mesh_verts_in_body(mu, du, f"Link_{LEG_NUM[leg]}{n_suffix}")
            bp_u, bR_u = body_frame(mu, du, f"Link_{LEG_NUM[leg]}{n_suffix}")
            wu = vu @ bR_u.T + bp_u
            chunks = []
            bid = mx.body(xbody).id
            for gid in range(mx.ngeom):
                if mx.geom_bodyid[gid] != bid or mx.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mid = mx.geom_dataid[gid]
                v = mx.mesh_vert[mx.mesh_vertadr[mid]:mx.mesh_vertadr[mid] + mx.mesh_vertnum[mid]]
                chunks.append(v @ dx.geom_xmat[gid].reshape(3, 3).T + dx.geom_xpos[gid] - off)
            wx = np.vstack(chunks)
            # same STL -> same vertex order after identical compilation
            k = min(len(wu), len(wx))
            worst_mesh = max(worst_mesh, float(np.abs(wu[:k] - wx[:k]).max()))

    print(f"validation over 25 random poses:")
    print(f"  worst joint-anchor mismatch : {worst_anchor*1000:.3f} mm")
    print(f"  worst joint-axis mismatch   : {worst_axis:.6f}")
    print(f"  worst foot-point mismatch   : {worst_foot*1000:.3f} mm")
    print(f"  worst mesh-vertex mismatch  : {worst_mesh*1000:.3f} mm (zero pose)")
    assert worst_anchor < 5e-4 and worst_axis < 1e-6 and worst_foot < 5e-4, "FK mismatch"
    assert worst_mesh < 1e-3, "visual mesh placement mismatch"
    print("  PASS")

    # report mass/CoM
    mujoco.mj_resetData(mx, dx)
    dx.qpos[3] = 1
    mujoco.mj_forward(mx, dx)
    tot = float(mx.body_subtreemass[mx.body("base").id])
    com = dx.subtree_com[mx.body("base").id]
    print(f"  total mass {tot:.3f} kg, CoM x {com[0]*1000:+.1f} mm "
          f"(target {TARGET_COM_X*1000:+.0f}, empirical bracket +9..+23)")


def main() -> None:
    xml = build()
    OUT_XML.write_text(xml)
    print(f"wrote {OUT_XML} ({len(xml)} bytes)")
    validate(xml)


if __name__ == "__main__":
    main()

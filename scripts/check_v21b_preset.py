"""Validation for the XGOLite-V21B preset (v21 rough-terrain stage).

Instantiates a small XGOLite-V21B env and asserts — by reading the BUILT
env back, not just the cfg — that the v21b terrain stack took effect:

1.  Terrain: generator active, 7 sub-terrain families, 10 difficulty rows x
    10 type columns, initial spawns on rows <= 2, stair heights obey the
    <= 0.5-leg-length rule (top-row stairs origin z consistent with 30 mm
    steps).
2.  BLIND actor: actor obs frame stays EXACTLY 49 dims (x history 5 = 245),
    no height_scan in the actor group; critic gets the 187-ray height_scan.
3.  Terrain-relative rewards: on a non-flat spawn (stairs column, top row)
    the scan-derived ground height matches the tile origin, the relative
    base height reads ~standing height while ABSOLUTE root z would be
    wrong, stance-foot relative heights are near zero, and both live
    reward terms are finite. Geometry is measured on crisp model-default
    contacts (the live compliance params are pinned benign for sections
    1-3): at full-severity draws the 0.4 s contact springs make the
    zero-action settle depth/drift vary run to run.
4.  Compliance DR (real params restored, forced full-severity redraw on
    the pinned top row): per-env solref/solimp on the foot pads differ
    across envs and sit inside the configured ranges; non-randomized
    entries (dampratio) stay at default.
5.  Slip event (at the top terrain row = full severity): a forced slip
    drops exactly one foot's friction into the slip range and a forced
    expiry restores the exact saved value; plus a functional MuJoCo
    max-mixing test — with priority-1 pads, a low foot friction WINS
    against the terrain-generator flat geom (low-mu robots slide much
    farther under an identical push).
5b. Hazard-DR severity coupling (v21b-v5): compliance + friction + slip
    events are wired severity_by_terrain_level / reset-mode; at s=0
    (row 0) contacts collapse to near-default (timeconst <= 0.05 s,
    solimp at the model default), friction never dips below 0.4 and
    slips fire as exact friction no-ops; at s=1 (top row) the draws
    cover exactly the old full-severity ranges.
6.  Terrain curriculum (v21b-v5.1: promotion/demotion on the honest
    achieved-velocity ratio, not reward fracs — the v5-run collapse fix):
    synthetic episodes with ADVERSARIAL fracs — ratio 0.6 promotes even
    where fracs would demote, 0.15 demotes even where fracs would
    promote, 0.4 holds; near-zero-command episode HOLDS (the low-speed
    demotion-coupling guard); a moving episode commanding neither ratio
    axis stays frac-gated; any commanded axis below the demote bar
    demotes; with the ratio params stripped the pre-v5.1 frac behavior is
    bit-identical; composes with the command grid (both terms present,
    grid still enabled).
6b. Grid unlock ratio AND-gate (v21b-v3, unlock_min_vel_ratio=0.5):
    achieved-velocity accumulators live and zeroed on reset; synthetic
    episodes — an IDLE episode (frac 0.9, achieved ~0, the v19-v5
    idle-inversion) must NOT unlock, a competent episode (ratio 0.9)
    must, reverse motion scores 0, a both-axes cell needs BOTH ratios,
    stand cells (neither axis commanded) stay frac-gated, and the
    per-cell MEAN ratio (competent + idle episode on one cell -> 0.45)
    blocks the unlock.
6c. Velocity-progress rewards (v21b-v5 anti-idle): both terms wired at
    the sized weights (>= +0.06/step at ratio 0.5, vs the measured
    ~ +0.03/step standing income); unit tests on a mocked env — zero at
    standing / reverse / uncommanded axes / orthogonal drift, the exact
    achieved ratio at synthetic velocities, clipped at 1 on overshoot.
7.  20 random-action steps produce finite rewards (smoke).
8.  V21A and V20 UNTOUCHED: plane terrain, no terrain sensors/events/
    curriculum, flat-task reward functions, original friction ranges and
    reset scatter, grid unlock_min_vel_ratio=None (ratio gate off, no
    accumulators), NO velocity-progress reward terms, stock startup
    dr.geom_friction (no severity params); both build and step finite,
    and the SAME synthetic idle episode that v21b's ratio gate blocks
    still unlocks there (pre-existing frac-only behavior bit-identical).

Usage:
  cd luwu_mjlab && PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python \
      scripts/check_v21b_preset.py
"""

import types

import torch

import mjlab.tasks  # noqa: F401  (registry)
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr as mjlab_dr
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity import mdp as local_mdp
from src.tasks.velocity.config.xgolite.env_cfgs import FOOT_PAD_GEOMS
from src.tasks.velocity.config.xgolite.v21b import (
  V21B_COMPLIANCE_BENIGN_TIMECONST,
  V21B_FRICTION_BENIGN_LOW,
  V21B_FRICTION_RANGE,
  V21B_GRID_MIN_VEL_RATIO,
  V21B_MAX_INIT_LEVEL,
  V21B_NUM_COLS,
  V21B_NUM_ROWS,
  V21B_PROGRESS_ANG_WEIGHT,
  V21B_PROGRESS_LIN_WEIGHT,
  V21B_SLIP_MU_RANGE,
  V21B_SOLIMP_D0_RANGE,
  V21B_SOLIMP_DMAX_RANGE,
  V21B_SOLIMP_WIDTH_RANGE,
  V21B_SOLREF_TIMECONST_RANGE,
  V21B_STEP_HEIGHT_RANGE,
  V21B_TERRAIN_DEMOTE_RATIO,
  V21B_TERRAIN_GAMMA_ANG,
  V21B_TERRAIN_GAMMA_LIN,
  V21B_TERRAIN_PROMOTE_RATIO,
)
from src.tasks.velocity.mdp.rewards import (
  feet_clearance_terrain,
  terrain_height_under_base,
  terrain_height_under_feet,
  track_base_height_terrain,
)

NUM_ENVS = 32
SMOKE_STEPS = 20
SEED = 0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Column layout of xgolite_v21b_terrain_gen_cfg (cumulative proportions):
COL_FLAT = 0          # cols 0-1 flat
COL_STAIRS = 3        # cols 3-4 pyramid stairs

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  status = "PASS" if ok else "FAIL"
  print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
  if not ok:
    _failures.append(name)


def pin_command(env: ManagerBasedRlEnv, vx: float, vy: float, wz: float) -> None:
  """Pin the twist command (check_v21a_preset pattern)."""
  term = env.command_manager.get_term("twist")
  term.cfg.rel_standing_envs = 0.0
  term.is_standing_env[:] = False

  def _pin(self, env_ids):
    self.vel_command_b[env_ids, 0] = vx
    self.vel_command_b[env_ids, 1] = vy
    self.vel_command_b[env_ids, 2] = wz
    if self.pose_enabled:
      self.vel_command_b[env_ids, 3] = 0.0
      self.vel_command_b[env_ids, 4] = 0.116
    self.is_standing_env[env_ids] = False
    if self.grid_enabled:
      self.grid_cell_index[env_ids] = -1

  term._resample_command = types.MethodType(_pin, term)


def pin_terrain(env: ManagerBasedRlEnv, row: int, col: int) -> None:
  """Place every env on one (difficulty row, type column) patch."""
  terrain = env.scene.terrain
  terrain.terrain_levels[:] = row
  terrain.terrain_types[:] = col
  terrain.env_origins[:] = terrain.terrain_origins[
    terrain.terrain_levels, terrain.terrain_types
  ]


def main() -> None:
  configure_torch_backends()
  torch.manual_seed(SEED)

  cfg = load_env_cfg("XGOLite-V21B")
  cfg.scene.num_envs = NUM_ENVS
  cfg.observations["actor"].enable_corruption = False
  env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE, render_mode=None)
  all_ids = torch.arange(NUM_ENVS, device=DEVICE)

  # Benign-contact override for the scan-GEOMETRY sections (1-3): with the
  # v21b-v5 reset-mode compliance DR, every reset redraws the foot-pad
  # solref/solimp, and at the top terrain row (s=1) the 0.4 s contact
  # springs let zero-action robots settle and drift by draw-varying
  # amounts — enough to push the scan-vs-origin medians across their
  # tolerances (observed 0.011 vs 0.030 on back-to-back runs). Geometry is
  # therefore measured on crisp model-default contacts: the manager's LIVE
  # resolved params are pinned to the default point here (degenerate
  # ranges draw exactly the default on every reset) and restored — with a
  # forced full-severity redraw — at the top of section 4, where the
  # severity machinery itself is under test.
  comp_term_cfg = env.event_manager.get_term_cfg("foot_compliance")
  orig_comp_params = dict(comp_term_cfg.params)
  comp_term_cfg.params.update(
    timeconst_range=(0.02, 0.02),
    solimp_d0_range=(0.9, 0.9),
    solimp_dmax_range=(0.95, 0.95),
    solimp_width_range=(0.001, 0.001),
    severity_by_terrain_level=False,
  )

  env.reset()
  zero_action = torch.zeros(
    NUM_ENVS, env.action_manager.total_action_dim, device=DEVICE
  )
  # One step initializes the class-based step events; then quiesce organic
  # slip events so the geometry measurements below are not perturbed (the
  # slip machinery is exercised explicitly in section 5).
  env.step(zero_action)
  slip_term = env.event_manager.get_term_cfg("foot_slip").func
  slip_term._time_to_next[:] = 1e9

  # ------------------------------------------------- 1. terrain grid ------
  terrain = env.scene.terrain
  gen_cfg = cfg.scene.terrain.terrain_generator
  check(
    "terrain generator active: 7 families, 10 rows x 10 cols",
    cfg.scene.terrain.terrain_type == "generator"
    and len(gen_cfg.sub_terrains) == 7
    and tuple(terrain.terrain_origins.shape[:2]) == (V21B_NUM_ROWS, V21B_NUM_COLS),
    f"origins {tuple(terrain.terrain_origins.shape)}",
  )
  init_max = int(terrain.terrain_levels.max())
  check(
    f"initial spawns limited to rows <= {V21B_MAX_INIT_LEVEL}",
    init_max <= V21B_MAX_INIT_LEVEL,
    f"max initial level {init_max}",
  )
  # Stairs top-row origin z ~ (num_steps + 1) * step_height with 6 steps at
  # difficulty in (0.9, 1.0): 7 * (0.0275..0.030) = 0.19..0.21 m.
  stairs_z = float(terrain.terrain_origins[V21B_NUM_ROWS - 1, COL_STAIRS, 2])
  lo = 7 * (
    V21B_STEP_HEIGHT_RANGE[0]
    + 0.9 * (V21B_STEP_HEIGHT_RANGE[1] - V21B_STEP_HEIGHT_RANGE[0])
  )
  check(
    "top-row stair column climbs per the <=0.5-leg-length step rule",
    lo - 0.02 <= stairs_z <= 7 * V21B_STEP_HEIGHT_RANGE[1] + 0.02,
    f"origin z {stairs_z:.3f} m (expected ~{lo:.3f}..{7 * V21B_STEP_HEIGHT_RANGE[1]:.3f})",
  )
  flat_z = float(terrain.terrain_origins[0, COL_FLAT, 2])
  check("flat column origin at z=0", abs(flat_z) < 1e-6, f"z {flat_z:.4f}")

  # -------------------------------------------- 2. blind obs contract -----
  om = env.observation_manager
  actor_dim = int(om.group_obs_dim["actor"][0])
  history = cfg.observations["actor"].history_length or 1
  check(
    "actor obs frame stays EXACTLY 49 dims (x history 5 = 245), blind",
    actor_dim == 245
    and history == 5
    and actor_dim // history == 49
    and "height_scan" not in cfg.observations["actor"].terms,
    f"flat dim {actor_dim}, history {history}, "
    f"actor terms {list(cfg.observations['actor'].terms)}",
  )
  critic_terms = cfg.observations["critic"].terms
  names = om.active_terms["critic"]
  dims = om.group_obs_term_dim["critic"]
  hs_dim = next(
    (int(d[0]) for n, d in zip(names, dims) if n == "height_scan"), None
  )
  check(
    "critic gets the privileged height_scan (187 rays)",
    "height_scan" in critic_terms
    and critic_terms["height_scan"].func is envs_mdp.height_scan
    and hs_dim == 187,
    f"height_scan dim {hs_dim}",
  )

  # ------------------------------- 3. terrain-relative rewards ------------
  pin_command(env, 0.0, 0.0, 0.0)
  pin_terrain(env, V21B_NUM_ROWS - 1, COL_STAIRS)
  torch.manual_seed(SEED + 3)  # pin the spawn-scatter draws for this reset
  env.reset()
  slip_term._time_to_next[:] = 1e9  # resets re-sample the slip timers
  for _ in range(75):  # settle (contacts pinned crisp by the override above)
    env.step(zero_action)
    # Robots that trip near the platform edge terminate and would be
    # DEMOTED by the live curriculum (working as intended); keep the
    # measurement population pinned to the top row.
    pin_terrain(env, V21B_NUM_ROWS - 1, COL_STAIRS)
  robot = env.scene["robot"]
  ground = terrain_height_under_base(env, "terrain_scan")
  origin_z = terrain.env_origins[:, 2]
  # Measure on upright, settled robots only: zero-action standing has no
  # balance controller, so spawns straddling a step edge fall over (and are
  # exactly what the curriculum demotes during training).
  ok_envs = (robot.data.projected_gravity_b[:, 2] < -0.95) & (
    env.episode_length_buf > 25
  )
  n_ok = int(ok_envs.sum())
  check(
    "enough upright settled robots on the stairs top row",
    n_ok >= 8,
    f"{n_ok}/{NUM_ENVS} upright",
  )
  ground_err = (ground - origin_z).abs()[ok_envs]
  # Robots scatter +/-0.25 m on a 0.6 m platform: most stand on the top
  # platform (ground == origin z), some straddle the first steps (one to
  # two 30 mm treads below). Use the median for the platform check.
  check(
    "scan ground height matches the stairs-top origin (non-flat spawn)",
    float(ground_err.median()) < 0.02 and float(ground_err.max()) < 0.12,
    f"median err {ground_err.median():.4f}, max {ground_err.max():.4f}, "
    f"origin z {origin_z[0]:.3f}",
  )
  rel_h = robot.data.root_link_pos_w[:, 2] - ground
  abs_err = (robot.data.root_link_pos_w[:, 2] - 0.116).abs()[ok_envs]
  rel_err = (rel_h - 0.116).abs()[ok_envs]
  check(
    "relative base height ~ standing height where absolute z is wrong",
    float(rel_err.median()) < 0.03 and float(abs_err.median()) > 0.05,
    f"median relative err {rel_err.median():.3f} vs absolute err "
    f"{abs_err.median():.3f}",
  )
  # Use the RESOLVED reward params (the manager deep-copies the cfg and
  # resolves SceneEntityCfg site names -> ids there, not on our cfg copy).
  fc_params = env.reward_manager.get_term_cfg("foot_clearance").params
  tbh_params = env.reward_manager.get_term_cfg("track_base_height").params
  site_ids = fc_params["asset_cfg"].site_ids
  foot_pos = robot.data.site_pos_w[:, site_ids]
  foot_rel = (
    foot_pos[..., 2] - terrain_height_under_feet(env, "terrain_scan", foot_pos)
  )[ok_envs]
  check(
    "stance-foot terrain-relative heights near zero on the stairs",
    float(foot_rel.abs().median()) < 0.03 and float(foot_rel.abs().max()) < 0.12,
    f"median |rel| {foot_rel.abs().median():.4f}, max {foot_rel.abs().max():.4f}",
  )
  r_height = track_base_height_terrain(env, **tbh_params)
  # The clearance cost is gated on the commanded twist; write a moving
  # command into the buffer so the gate is open (robot stays standing —
  # the cost is velocity-weighted, so it must come out small but live).
  cmd_buf = env.command_manager.get_term("twist").vel_command_b
  cmd_buf[:, 0] = 0.2
  r_clear = feet_clearance_terrain(env, **fc_params)
  cmd_buf[:, 0] = 0.0
  check(
    "live terrain-relative reward terms finite and sane",
    bool(torch.isfinite(r_height).all())
    and bool(torch.isfinite(r_clear).all())
    and float(r_height[ok_envs].median()) > 0.05
    and float(r_clear.max()) < 1.0,
    f"height reward median (upright) {r_height[ok_envs].median():.3f}, "
    f"clearance cost range [{r_clear.min():.4f}, {r_clear.max():.4f}]",
  )
  check(
    "reward table wires the terrain-relative functions",
    cfg.rewards["track_base_height"].func is local_mdp.track_base_height_terrain
    and cfg.rewards["foot_clearance"].func is local_mdp.feet_clearance_terrain,
  )

  # -------------------------------------------- 4. compliance DR ----------
  # Restore the real compliance params (geometry override above) and force
  # a redraw: the robots still sit on the pinned top row (s=1), so the
  # draws below must cover the full configured ranges.
  comp_term_cfg.params.clear()
  comp_term_cfg.params.update(orig_comp_params)
  torch.manual_seed(SEED + 4)
  local_mdp.foot_contact_compliance(env, all_ids, **comp_term_cfg.params)
  foot_gids = slip_term._geom_ids
  model = env.sim.model
  solref = model.geom_solref[:, foot_gids, :]
  solimp = model.geom_solimp[:, foot_gids, :]
  tc = solref[..., 0]
  check(
    "compliance DR wrote per-env solref timeconst (varies, in range)",
    float(tc.std(dim=0).mean()) > 0.01
    and float(tc.min()) >= V21B_SOLREF_TIMECONST_RANGE[0] - 1e-6
    and float(tc.max()) <= V21B_SOLREF_TIMECONST_RANGE[1] + 1e-6
    and bool((solref[..., 1] == solref[0, 0, 1]).all()),  # dampratio untouched
    f"timeconst range [{tc.min():.3f}, {tc.max():.3f}], "
    f"across-env std {tc.std(dim=0).mean():.3f}",
  )
  d0, dmax, width = solimp[..., 0], solimp[..., 1], solimp[..., 2]
  check(
    "compliance DR wrote per-env solimp (d0 < dmax, all in range)",
    bool((d0 < dmax).all())
    and float(d0.min()) >= V21B_SOLIMP_D0_RANGE[0] - 1e-6
    and float(dmax.max()) <= V21B_SOLIMP_DMAX_RANGE[1] + 1e-6
    and float(width.min()) >= V21B_SOLIMP_WIDTH_RANGE[0] - 1e-6
    and float(width.max()) <= V21B_SOLIMP_WIDTH_RANGE[1] + 1e-6
    and float(dmax.std(dim=0).mean()) > 0.005,
    f"d0 [{d0.min():.3f}, {d0.max():.3f}], dmax [{dmax.min():.3f}, "
    f"{dmax.max():.3f}], width [{width.min():.4f}, {width.max():.4f}]",
  )

  # ------------------------------------------------ 5. slip event ---------
  # Move to the flat column on settled robots first: env resets CANCEL an
  # active slip (by design), so forcing slips while stair-spawned robots
  # are still terminating would immediately undo some of them. TOP row:
  # the flat column is identical on every row, and at severity s=1 the
  # friction/slip draws must reproduce the pre-severity full ranges
  # exactly (row-0 benign behavior is section 5b).
  pin_terrain(env, V21B_NUM_ROWS - 1, COL_FLAT)
  torch.manual_seed(SEED + 5)  # pin the friction/scatter draws (the RNG
  # position otherwise drifts with the run-varying termination count above)
  env.reset()
  slip_term._time_to_next[:] = 1e9  # resets re-sample the slip timers
  for _ in range(30):
    env.step(zero_action)
  if bool(slip_term._active.any()):
    slip_term._slip_left[:] = 0.0
    env.step(zero_action)
  base_mu = model.geom_friction[
    torch.arange(NUM_ENVS, device=DEVICE)[:, None], foot_gids, 0
  ].clone()
  check(
    "per-env foot friction drawn from the widened (0.05, 2.0) range",
    float(base_mu.min()) >= V21B_FRICTION_RANGE[0] - 1e-6
    and float(base_mu.max()) <= V21B_FRICTION_RANGE[1] + 1e-6
    and float(base_mu[:, 0].std()) > 0.1,
    f"mu range [{base_mu.min():.3f}, {base_mu.max():.3f}]",
  )
  slip_term._time_to_next[:] = 0.0
  env.step(zero_action)
  slipped_gid = foot_gids[slip_term._slip_foot]
  mu_now = model.geom_friction[all_ids, slipped_gid, 0]
  active = slip_term._active.clone()
  check(
    "forced slip: every env has one foot dropped into the slip mu range",
    bool(active.all())
    and float(mu_now.max()) <= V21B_SLIP_MU_RANGE[1] + 1e-6
    and float(mu_now.min()) >= V21B_SLIP_MU_RANGE[0] - 1e-6,
    f"slipped mu range [{mu_now.min():.3f}, {mu_now.max():.3f}]",
  )
  other = model.geom_friction[
    all_ids[:, None], foot_gids, 0
  ]  # [B, 4] incl. slipped
  n_low = (other <= V21B_SLIP_MU_RANGE[1] + 1e-6).sum(dim=1)
  base_low = (base_mu <= V21B_SLIP_MU_RANGE[1] + 1e-6).sum(dim=1)
  check(
    "slip affects exactly ONE foot (others keep the startup draw)",
    bool((n_low - base_low <= 1).all()),
    f"newly-low feet per env max {(n_low - base_low).max()}",
  )
  slip_term._slip_left[:] = 0.0
  env.step(zero_action)
  mu_restored = model.geom_friction[all_ids, slipped_gid, 0]
  saved = base_mu[all_ids, slip_term._slip_foot]
  check(
    "slip expiry restores the exact pre-slip friction",
    bool(~slip_term._active.any())
    and bool(torch.allclose(mu_restored, saved, atol=1e-6)),
    f"max restore err {(mu_restored - saved).abs().max():.2e}",
  )

  # Functional max-mixing test: on the terrain-generator FLAT geom
  # (default friction 1.0), the priority-1 foot pads' LOW friction must
  # win the contact pairing — under an identical base push, low-mu FEET
  # must slide while high-mu feet stay planted (the base displacement is a
  # bad proxy on this kp-5.6 soft plant: it leans several cm without any
  # foot slip). If MuJoCo's max-mixing applied (no priority), both groups
  # would see contact mu 1.0 and neither would slide.
  low = all_ids[: NUM_ENVS // 2]
  high = all_ids[NUM_ENVS // 2 :]
  env_grid = all_ids[:, None]
  model.geom_friction[env_grid[low], foot_gids, 0] = 0.02
  model.geom_friction[env_grid[high], foot_gids, 0] = 1.5
  foot_start = robot.data.site_pos_w[:, site_ids, :2].clone()
  envs_mdp.push_by_setting_velocity(
    env, all_ids, velocity_range={"x": (0.3, 0.3)}
  )
  peak = torch.zeros(NUM_ENVS, device=DEVICE)
  alive = torch.ones(NUM_ENVS, dtype=torch.bool, device=DEVICE)
  count = env.episode_length_buf.clone()
  for _ in range(15):  # 0.3 s
    env.step(zero_action)
    # Envs that reset (tripped and terminated) teleport back to the spawn
    # scatter — their displacement is meaningless, exclude them.
    alive &= env.episode_length_buf > count
    count = env.episode_length_buf.clone()
    foot_disp = torch.norm(
      robot.data.site_pos_w[:, site_ids, :2] - foot_start, dim=-1
    ).mean(dim=1)
    peak = torch.maximum(peak, torch.where(alive, foot_disp, peak))
  low_alive, high_alive = alive[low], alive[high]
  low_d = float(peak[low][low_alive].median()) if low_alive.any() else float("nan")
  high_d = (
    float(peak[high][high_alive].median()) if high_alive.any() else float("nan")
  )
  check(
    "max-mixing handled: low foot mu WINS vs terrain geom (feet slide)",
    int(low_alive.sum()) >= 4
    and int(high_alive.sum()) >= 4
    and low_d > 2.0 * high_d
    and low_d > 0.02,
    f"median foot slide 0.3 s after identical 0.3 m/s push: mu=0.02 -> "
    f"{low_d:.3f} m ({int(low_alive.sum())} alive), mu=1.5 -> {high_d:.3f} m "
    f"({int(high_alive.sum())} alive)",
  )

  # ---------------------- 5b. terrain-level DR severity coupling ----------
  ev = env.event_manager
  check(
    "severity coupling wired (reset-mode compliance/friction, slip flags)",
    cfg.events["foot_compliance"].mode == "reset"
    and cfg.events["foot_compliance"].params["severity_by_terrain_level"] is True
    and cfg.events["foot_friction"].mode == "reset"
    and cfg.events["foot_friction"].func
    is local_mdp.foot_friction_terrain_scaled
    and cfg.events["foot_friction"].params["benign_low"]
    == V21B_FRICTION_BENIGN_LOW
    and cfg.events["foot_slip"].params["severity_by_terrain_level"] is True
    and cfg.events["foot_slip"].params["restore_on_reset"] is False,
    f"compliance mode {cfg.events['foot_compliance'].mode}, "
    f"friction mode {cfg.events['foot_friction'].mode}",
  )
  # Settle on the flat column first (the push test above left robots
  # tripping; resets during the steps below would redraw friction and
  # poison the no-op comparison).
  pin_terrain(env, 0, COL_FLAT)
  torch.manual_seed(SEED + 55)
  env.reset()
  slip_term._time_to_next[:] = 1e9
  for _ in range(10):
    env.step(zero_action)
  # Use the manager-RESOLVED params (asset_cfg geom names -> ids).
  comp_params = ev.get_term_cfg("foot_compliance").params
  fric_params = ev.get_term_cfg("foot_friction").params

  def redraw_stats(n_draws: int) -> dict[str, torch.Tensor]:
    tc, d0s, dmaxs, widths, mus = [], [], [], [], []
    for _ in range(n_draws):
      local_mdp.foot_contact_compliance(env, all_ids, **comp_params)
      local_mdp.foot_friction_terrain_scaled(env, all_ids, **fric_params)
      tc.append(model.geom_solref[:, foot_gids, 0].clone())
      d0s.append(model.geom_solimp[:, foot_gids, 0].clone())
      dmaxs.append(model.geom_solimp[:, foot_gids, 1].clone())
      widths.append(model.geom_solimp[:, foot_gids, 2].clone())
      mus.append(model.geom_friction[all_ids[:, None], foot_gids, 0].clone())
    return {
      "tc": torch.stack(tc),
      "d0": torch.stack(d0s),
      "dmax": torch.stack(dmaxs),
      "width": torch.stack(widths),
      "mu": torch.stack(mus),
    }

  # s = 0 (row 0): benign physics — near-default contacts, no ice.
  terrain.terrain_levels[:] = 0
  s0 = redraw_stats(5)
  check(
    "s=0: compliance benign (timeconst <= 0.05 s, solimp = model default)",
    float(s0["tc"].max()) <= V21B_COMPLIANCE_BENIGN_TIMECONST + 1e-6
    and float(s0["tc"].min()) >= V21B_SOLREF_TIMECONST_RANGE[0] - 1e-6
    and float((s0["d0"] - 0.9).abs().max()) < 1e-6
    and float((s0["dmax"] - 0.95).abs().max()) < 1e-6
    and float((s0["width"] - 0.001).abs().max()) < 1e-7,
    f"tc [{s0['tc'].min():.4f}, {s0['tc'].max():.4f}], "
    f"d0 max dev {(s0['d0'] - 0.9).abs().max():.2e}",
  )
  check(
    "s=0: friction low end >= 0.4 (no ice on row 0)",
    float(s0["mu"].min()) >= V21B_FRICTION_BENIGN_LOW - 1e-6
    and float(s0["mu"].max()) <= V21B_FRICTION_RANGE[1] + 1e-6,
    f"mu [{s0['mu'].min():.3f}, {s0['mu'].max():.3f}]",
  )
  # s=0 slip: fires but is an exact friction no-op.
  pre = model.geom_friction[all_ids[:, None], foot_gids, 0].clone()
  count0 = env.episode_length_buf.clone()
  slip_term._time_to_next[:] = 0.0
  env.step(zero_action)
  stayed = env.episode_length_buf > count0  # reset envs redraw; exclude
  post = model.geom_friction[all_ids[:, None], foot_gids, 0]
  check(
    "s=0: slip events fire as friction no-ops",
    int(stayed.sum()) >= 8
    and bool(slip_term._active[stayed].all())
    and bool(torch.allclose(post[stayed], pre[stayed], atol=1e-6)),
    f"{int(stayed.sum())}/{NUM_ENVS} non-reset envs, max delta "
    f"{(post[stayed] - pre[stayed]).abs().max():.2e}",
  )
  slip_term._slip_left[:] = 0.0
  env.step(zero_action)
  slip_term._time_to_next[:] = 1e9

  # s = 1 (top row): the draws must cover exactly the old full ranges.
  terrain.terrain_levels[:] = V21B_NUM_ROWS - 1
  s1 = redraw_stats(10)  # 10 x 32 envs x 4 pads: range-reach is robust
  check(
    "s=1: compliance draws span the full configured ranges",
    float(s1["tc"].min()) >= V21B_SOLREF_TIMECONST_RANGE[0] - 1e-6
    and float(s1["tc"].max()) <= V21B_SOLREF_TIMECONST_RANGE[1] + 1e-6
    and float(s1["tc"].max()) > 0.3  # reaches beyond any benign ceiling
    and float(s1["d0"].min()) >= V21B_SOLIMP_D0_RANGE[0] - 1e-6
    and float(s1["d0"].min()) < 0.7
    and float(s1["dmax"].max()) <= V21B_SOLIMP_DMAX_RANGE[1] + 1e-6
    and float(s1["width"].max()) <= V21B_SOLIMP_WIDTH_RANGE[1] + 1e-6
    and float(s1["width"].max()) > 0.005,
    f"tc [{s1['tc'].min():.3f}, {s1['tc'].max():.3f}], "
    f"d0 min {s1['d0'].min():.3f}, width max {s1['width'].max():.4f}",
  )
  check(
    "s=1: friction draws recover the full (0.05, 2.0) ice tail",
    float(s1["mu"].min()) >= V21B_FRICTION_RANGE[0] - 1e-6
    and float(s1["mu"].max()) <= V21B_FRICTION_RANGE[1] + 1e-6
    and float(s1["mu"].min()) < 0.15,  # dips well below the benign 0.4
    f"mu [{s1['mu'].min():.3f}, {s1['mu'].max():.3f}]",
  )

  # ------------------------------------------- 6. terrain curriculum ------
  curr_params = cfg.curriculum["terrain_levels"].params
  check(
    "terrain curriculum composes with the command grid",
    "command_grid" in cfg.curriculum
    and env.command_manager.get_term("twist").grid_enabled
    and curr_params["gamma_lin"] == V21B_TERRAIN_GAMMA_LIN
    and curr_params["gamma_ang"] == V21B_TERRAIN_GAMMA_ANG,
    f"curriculum terms {list(cfg.curriculum)}",
  )
  # v21b-v5.1: promotion gates on the honest achieved-velocity ratio (the
  # v5 run's frac gates scored standing ~0.49 > honest walking ~0.26 and
  # demoted every env to row 0 by iter 500 — see v21b.py point 11).
  check(
    "terrain promotion gates on the honest velocity ratio (0.5 / 0.25)",
    curr_params["promote_min_vel_ratio"] == V21B_TERRAIN_PROMOTE_RATIO
    and curr_params["demote_below_vel_ratio"] == V21B_TERRAIN_DEMOTE_RATIO,
    f"promote {curr_params['promote_min_vel_ratio']}, "
    f"demote {curr_params['demote_below_vel_ratio']}",
  )
  rm = env.reward_manager
  dt_scale = env.step_dt if rm._scale_by_dt else 1.0
  w_lin = rm.get_term_cfg("track_linear_velocity").weight
  w_ang = rm.get_term_cfg("track_angular_velocity").weight
  gterm6 = env.command_manager.get_term("twist")
  cmd = gterm6.vel_command_b
  steps = float(env.max_episode_length)

  def terrain_episode(
    i: int,
    cmd_vec: tuple[float, float, float],
    frac: float,
    ach_vx: float,
    ach_wz: float,
  ) -> None:
    """Fabricate one ended episode: command, reward fracs, achieved sums."""
    env.episode_length_buf[i] = int(steps)
    cmd[i, 0], cmd[i, 1], cmd[i, 2] = cmd_vec
    rm._episode_sums["track_linear_velocity"][i] = steps * w_lin * dt_scale * frac
    rm._episode_sums["track_angular_velocity"][i] = steps * w_ang * dt_scale * frac
    gterm6.grid_achieved_lin_sum[i, 0] = ach_vx * steps
    gterm6.grid_achieved_lin_sum[i, 1] = 0.0
    gterm6.grid_achieved_ang_sum[i] = ach_wz * steps

  # Ratio-mode synthetic episodes on envs 0..5, all starting at level 3.
  # Fracs are set ADVERSARIALLY (opposite frac-gate decision) to prove the
  # ratio, not the frac, decides on ratio-evaluable episodes.
  ids = torch.tensor([0, 1, 2, 3, 4, 5], device=DEVICE)
  terrain.terrain_levels[ids] = 3
  # env 0: ratio 0.6 >= 0.5, frac 0.2 (frac alone would DEMOTE) -> PROMOTE.
  terrain_episode(0, (0.2, 0.0, 0.0), 0.2, 0.6 * 0.2, 0.0)
  # env 1: near-zero command -> HOLD (moving guard, unchanged).
  terrain_episode(1, (0.0, 0.0, 0.0), 0.0, 0.0, 0.0)
  # env 2: ratio 0.15 < 0.25, frac 0.9 (frac alone would PROMOTE) -> DEMOTE.
  terrain_episode(2, (0.2, 0.0, 0.0), 0.9, 0.15 * 0.2, 0.0)
  # env 3: ratio 0.4 in [0.25, 0.5), frac 0.9 -> HOLD.
  terrain_episode(3, (0.2, 0.0, 0.0), 0.9, 0.4 * 0.2, 0.0)
  # env 4: moving (norm 0.11) but NEITHER axis ratio-commanded (0.04 < 0.05,
  # 0.07 < 0.1): ratio-exempt, frac 0.9 gates -> PROMOTE (stand-adjacent
  # episodes keep the frac behavior).
  terrain_episode(4, (0.04, 0.0, 0.07), 0.9, 0.0, 0.0)
  # env 5: both axes commanded; lin ratio 0.6 passes but ang ratio 0.15
  # fails the demote bar -> DEMOTE (any commanded axis below 0.25).
  terrain_episode(5, (0.2, 0.0, 0.5), 0.9, 0.6 * 0.2, 0.15 * 0.5)
  local_mdp.terrain_levels_reward_gated(env, ids, **curr_params)
  levels = terrain.terrain_levels[ids].tolist()
  check(
    "ratio-gated terrain curriculum: promote 0.6 / hold 0.4 / demote 0.15, "
    "frac-exempt stand-adjacent, any-axis demote",
    levels == [4, 3, 2, 3, 4, 2],
    f"levels after synthetic episodes {levels} (want [4, 3, 2, 3, 4, 2])",
  )
  origins_ok = bool(
    torch.allclose(
      terrain.env_origins[ids],
      terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]],
    )
  )
  check("curriculum moved the env origins with the levels", origins_ok)
  # Default-off (ratio params stripped) = the pre-v5.1 frac behavior
  # bit-identical: the same fabricated fracs decide, achieved sums ignored.
  frac_params = {k: v for k, v in curr_params.items() if "vel_ratio" not in k}
  ids3 = torch.tensor([0, 1, 2], device=DEVICE)
  terrain.terrain_levels[ids3] = 3
  terrain_episode(0, (0.2, 0.0, 0.0), 0.9, 0.0, 0.0)  # frac promote, ratio 0
  terrain_episode(1, (0.0, 0.0, 0.0), 0.0, 0.0, 0.0)  # stand hold
  terrain_episode(2, (0.2, 0.0, 0.0), 0.1, 0.9 * 0.2, 0.0)  # frac demote
  local_mdp.terrain_levels_reward_gated(env, ids3, **frac_params)
  levels3 = terrain.terrain_levels[ids3].tolist()
  check(
    "terrain curriculum defaults (no ratio params) keep frac gating",
    levels3 == [4, 3, 2],
    f"levels {levels3} (want [4, 3, 2])",
  )

  # ------------------------------ 6b. grid unlock ratio AND-gate ----------
  gterm = env.command_manager.get_term("twist")
  gc = gterm.cfg.grid_curriculum
  check(
    "V21B opts into the achieved-velocity ratio gate",
    gc is not None
    and gc.unlock_min_vel_ratio == V21B_GRID_MIN_VEL_RATIO
    and gterm.grid_ratio_enabled,
    f"unlock_min_vel_ratio={getattr(gc, 'unlock_min_vel_ratio', None)}",
  )
  # Accumulators wired: stepping accumulates, env reset zeroes.
  snap = gterm.grid_achieved_lin_sum.clone()
  env.step(zero_action)
  env.step(zero_action)
  check(
    "achieved-velocity accumulator integrates during stepping",
    not torch.equal(snap, gterm.grid_achieved_lin_sum),
  )
  env.reset()
  slip_term._time_to_next[:] = 1e9
  check(
    "achieved-velocity accumulators zeroed on episode reset",
    bool((gterm.grid_achieved_lin_sum == 0).all())
    and bool((gterm.grid_achieved_ang_sum == 0).all()),
  )

  # Synthetic ratio-gate episodes. Grid inherited from v18range: 21 x 8
  # cells over vx (-0.45, 0.60) x wz (-1.0, 1.0), seed vx [-0.15, 0.25] x
  # wz [-0.5, 0.5]. Cell A (13, 4) = vx [0.20, 0.25] x wz [0.0, 0.25]
  # sits on the seed edge; its +vx neighbor (14, 4) starts locked at 0.
  n_wz_g = gterm._grid_n_wz
  check(
    "grid geometry is the inherited 21 x 8",
    (gterm._grid_n_vx, n_wz_g) == (21, 8),
    f"{gterm._grid_n_vx} x {n_wz_g}",
  )
  g_steps = 200

  def grid_episode(
    i: int,
    cell: int,
    cmd_vec: tuple[float, float, float],
    frac: float,
    ach_xy: tuple[float, float],
    ach_wz: float,
  ) -> None:
    env.episode_length_buf[i] = g_steps
    rm._episode_sums["track_linear_velocity"][i] = frac * g_steps * dt_scale * w_lin
    rm._episode_sums["track_angular_velocity"][i] = frac * g_steps * dt_scale * w_ang
    gterm.grid_cell_index[i] = cell
    gterm.vel_command_b[i, 0] = cmd_vec[0]
    gterm.vel_command_b[i, 1] = cmd_vec[1]
    gterm.vel_command_b[i, 2] = cmd_vec[2]
    gterm.grid_achieved_lin_sum[i, 0] = ach_xy[0] * g_steps
    gterm.grid_achieved_lin_sum[i, 1] = ach_xy[1] * g_steps
    gterm.grid_achieved_ang_sum[i] = ach_wz * g_steps

  def run_grid(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    gterm.grid_weights.copy_(gterm.grid_seed_mask.float())
    before = gterm.grid_weights.clone()
    gids = torch.arange(n, device=DEVICE)
    other = torch.arange(n, NUM_ENVS, device=DEVICE)
    gterm.grid_cell_index[other] = -1  # only the synthetic envs attribute
    local_mdp.command_grid_adaptive(env, gids, command_name="twist")
    return before, gterm.grid_weights

  cell_a = 13 * n_wz_g + 4  # cmd (0.225, 0, 0.05): lin commanded, ang not
  cell_b = 13 * n_wz_g + 5  # cmd (0.225, 0, 0.30): BOTH axes commanded

  # (a) IDLE episode: high frac (the v19-v5 idle-inversion), achieved ~0.
  grid_episode(0, cell_a, (0.225, 0.0, 0.05), 0.9, (0.0, 0.0), 0.0)
  before, after = run_grid(1)
  check(
    "idle episode (frac 0.9, achieved ratio 0) does NOT unlock",
    bool(torch.equal(before, after)),
    f"(14,4): {after[14, 4].item():.2f} (want 0.00)",
  )
  # (b) competent episode: achieved 0.9 x command -> unlocks the neighbor.
  grid_episode(0, cell_a, (0.225, 0.0, 0.05), 0.9, (0.9 * 0.225, 0.0), 0.0)
  before, after = run_grid(1)
  expected = before.clone()
  expected[14, 4] = 0.2
  check(
    "competent episode (ratio 0.9) unlocks (+0.2 on the locked neighbor)",
    bool(torch.allclose(after, expected)),
    f"(14,4): {after[14, 4].item():.2f} (want 0.20)",
  )
  # (c) reverse motion is directional: ratio clips to 0, no unlock.
  grid_episode(0, cell_a, (0.225, 0.0, 0.05), 0.9, (-0.9 * 0.225, 0.0), 0.0)
  before, after = run_grid(1)
  check(
    "reverse-motion episode (ratio clipped to 0) does NOT unlock",
    bool(torch.equal(before, after)),
  )
  # (d) both-axes cell: lin ratio alone is not enough, lin+ang unlocks.
  grid_episode(0, cell_b, (0.225, 0.0, 0.30), 0.9, (0.9 * 0.225, 0.0), 0.0)
  before, after = run_grid(1)
  lin_only_blocked = bool(torch.equal(before, after))
  grid_episode(0, cell_b, (0.225, 0.0, 0.30), 0.9, (0.9 * 0.225, 0.0), 0.9 * 0.30)
  before, after = run_grid(1)
  check(
    "both-axes cell: lin ratio alone blocked, lin+ang ratios unlock",
    lin_only_blocked
    and bool((after != before).any())
    and abs(after[14, 5].item() - 0.2) < 1e-6,
    f"lin-only blocked: {lin_only_blocked}, (14,5): {after[14, 5].item():.2f}",
  )
  # (e) stand cell (|cmd_xy| < 0.05, |wz| < 0.1 but twist norm > 0.05):
  # exempt from the ratio gate — frac gates alone unlock. Weights are
  # zeroed first so the +0.2 bumps are visible on the seed cells too.
  cell_stand = 9 * n_wz_g + 4  # vx [0.0, 0.05) x wz [0.0, 0.25); cmd wz 0.08
  grid_episode(0, cell_stand, (0.0, 0.0, 0.08), 0.9, (0.0, 0.0), 0.0)
  gterm.grid_weights.zero_()
  gids = torch.arange(1, device=DEVICE)
  gterm.grid_cell_index[torch.arange(1, NUM_ENVS, device=DEVICE)] = -1
  local_mdp.command_grid_adaptive(env, gids, command_name="twist")
  stand_w = gterm.grid_weights
  check(
    "stand cell (no commanded axis) stays frac-gated (ratio exempt)",
    abs(stand_w.sum().item() - 1.0) < 1e-6
    and abs(stand_w[9, 4].item() - 0.2) < 1e-6,
    f"bumped weight sum {stand_w.sum().item():.2f} (want 1.00 = 5 x 0.2), "
    f"(9,4): {stand_w[9, 4].item():.2f}",
  )
  # (f) per-cell MEAN: competent (0.9) + idle (0.0) episode on the same
  # cell -> mean ratio 0.45 < 0.5, the cell must NOT unlock even though
  # one episode passed the frac gates.
  grid_episode(0, cell_a, (0.225, 0.0, 0.05), 0.9, (0.9 * 0.225, 0.0), 0.0)
  grid_episode(1, cell_a, (0.225, 0.0, 0.05), 0.9, (0.0, 0.0), 0.0)
  before, after = run_grid(2)
  check(
    "cell MEAN ratio gates the unlock (0.9 + 0.0 episodes -> 0.45 < 0.5)",
    bool(torch.equal(before, after)),
    f"(14,4): {after[14, 4].item():.2f} (want 0.00)",
  )
  # Restore a clean grid state for the smoke section.
  gterm.grid_weights.copy_(gterm.grid_seed_mask.float())
  rm._episode_sums["track_linear_velocity"][:] = 0.0
  rm._episode_sums["track_angular_velocity"][:] = 0.0
  gterm.grid_achieved_lin_sum.zero_()
  gterm.grid_achieved_ang_sum.zero_()
  gterm.grid_cell_index[:] = -1
  env.episode_length_buf[:] = 0

  # ------------------------------ 6c. velocity-progress rewards -----------
  pl_cfg = cfg.rewards["velocity_progress_lin"]
  pa_cfg = cfg.rewards["velocity_progress_ang"]
  # Sizing contract (v21b.py wiring comment): contribution = raw * weight *
  # step_dt; ratio-0.5 walking must add >= +0.06/step (2x the measured
  # ~ +0.03/step standing income, whose progress contribution is 0).
  lin_05 = pl_cfg.weight * 0.5 * dt_scale
  ang_05 = pa_cfg.weight * 0.5 * dt_scale
  check(
    "progress terms wired at sized weights (>= +0.06/step at ratio 0.5)",
    pl_cfg.func is local_mdp.velocity_progress_lin
    and pa_cfg.func is local_mdp.velocity_progress_ang
    and pl_cfg.weight == V21B_PROGRESS_LIN_WEIGHT
    and pa_cfg.weight == V21B_PROGRESS_ANG_WEIGHT
    and lin_05 >= 0.06 - 1e-9
    and ang_05 >= 0.06 - 1e-9,
    f"lin w {pl_cfg.weight} -> +{lin_05:.3f}/step at ratio 0.5 "
    f"(+{2 * lin_05:.3f} at 1.0), ang w {pa_cfg.weight} -> +{ang_05:.3f}",
  )
  # Unit tests on a mocked env (same duck-typed surface the terms read:
  # scene[name].data velocities + command_manager.get_command).
  prog_cmd = torch.tensor(
    [
      # (cmd_vx, cmd_vy, cmd_wz)
      [0.20, 0.00, 0.50],   # standing under a move command
      [0.20, 0.00, 0.50],   # full tracking
      [0.20, 0.00, 0.50],   # half tracking
      [0.20, 0.00, 0.50],   # reverse motion (directional -> 0)
      [0.20, 0.00, 0.50],   # overshoot (clips at 1)
      [0.04, 0.00, 0.05],   # both axes below the command gates
      [0.20, 0.00, 0.50],   # orthogonal drift (no progress along cmd)
      [0.10, 0.10, -0.60],  # diagonal cmd; negative yaw tracked at half
    ]
  )
  prog_vel = torch.tensor(
    [
      # (vx_b, vy_b, wz_b)
      [0.00, 0.00, 0.00],
      [0.20, 0.00, 0.50],
      [0.10, 0.00, 0.25],
      [-0.20, 0.00, -0.50],
      [0.40, 0.00, 1.20],
      [0.50, 0.00, 1.00],
      [0.00, 0.30, 0.00],
      [0.05, 0.05, -0.30],
    ]
  )
  lin_vel_b = torch.zeros(len(prog_vel), 3)
  lin_vel_b[:, :2] = prog_vel[:, :2]
  ang_vel_b = torch.zeros(len(prog_vel), 3)
  ang_vel_b[:, 2] = prog_vel[:, 2]
  mock_env = types.SimpleNamespace(
    scene={
      "robot": types.SimpleNamespace(
        data=types.SimpleNamespace(
          root_link_lin_vel_b=lin_vel_b, root_link_ang_vel_b=ang_vel_b
        )
      )
    },
    command_manager=types.SimpleNamespace(get_command=lambda name: prog_cmd),
  )
  r_lin = local_mdp.velocity_progress_lin(mock_env, command_name="twist")
  r_ang = local_mdp.velocity_progress_ang(mock_env, command_name="twist")
  want = torch.tensor([0.0, 1.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5])
  check(
    "progress rewards: 0 at stand/reverse/uncommanded, exact ratio else",
    bool(torch.allclose(r_lin, want, atol=1e-6))
    and bool(torch.allclose(r_ang, want, atol=1e-6)),
    f"lin {[round(v, 3) for v in r_lin.tolist()]}, "
    f"ang {[round(v, 3) for v in r_ang.tolist()]}",
  )
  # Live wiring: both terms are active in the built reward manager.
  check(
    "progress terms active in the built reward manager",
    "velocity_progress_lin" in rm.active_terms
    and "velocity_progress_ang" in rm.active_terms,
  )

  # --------------------------------------- 7. random-action smoke ---------
  pin_command(env, 0.2, 0.05, 0.0)
  torch.manual_seed(SEED + 7)
  env.reset()
  slip_term._time_to_next[:] = 0.05  # re-arm organic slips inside the smoke
  action = torch.zeros_like(zero_action)
  finite = True
  reward_sum = 0.0
  for _ in range(SMOKE_STEPS):
    action.uniform_(-1.0, 1.0)
    _, reward, terminated, truncated, _ = env.step(action)
    finite = (
      finite
      and bool(torch.isfinite(reward).all())
      and bool(torch.isfinite(robot.data.joint_pos).all())
      and bool(torch.isfinite(robot.data.joint_vel).all())
    )
    reward_sum += float(reward.mean())
  check(
    f"{SMOKE_STEPS} random-action steps finite on terrain (no NaN)",
    finite,
    f"mean step reward {reward_sum / SMOKE_STEPS:.4f}",
  )
  env.close()
  del env

  # ------------------------------------------- 8. V21A / V20 untouched ----
  for task in ("XGOLite-V21A", "XGOLite-V20"):
    c = load_env_cfg(task)
    tbh = c.rewards["track_base_height"]
    fcl = c.rewards["foot_clearance"]
    check(
      f"{task} cfg untouched (plane, no terrain stack, flat rewards)",
      c.scene.terrain.terrain_type == "plane"
      and c.scene.terrain.terrain_generator is None
      and all(s.name != "terrain_scan" for s in (c.scene.sensors or ()))
      and "height_scan" not in c.observations["actor"].terms
      and "height_scan" not in c.observations["critic"].terms
      and "foot_compliance" not in c.events
      and "foot_slip" not in c.events
      and "terrain_levels" not in c.curriculum
      and c.events["foot_friction"].params["ranges"] == (0.25, 2.0)
      and tbh.func is local_mdp.track_base_height
      and "sensor_name" not in tbh.params
      # NOTE: the base factory's `mdp` alias is shadowed by the local mdp
      # import (velocity_env_cfg.py line 33), so flat tasks resolve
      # feet_clearance from the LOCAL module.
      and fcl.func is local_mdp.feet_clearance
      and "sensor_name" not in fcl.params
      and c.events["reset_base"].params["pose_range"]["x"] == (-0.5, 0.5)
      and c.events["reset_base"].params["pose_range"]["z"] == (0.0, 0.0),
      f"terrain {c.scene.terrain.terrain_type}, "
      f"friction {c.events['foot_friction'].params['ranges']}",
    )
    check(
      f"{task} grid keeps unlock_min_vel_ratio=None (ratio gate off)",
      c.commands["twist"].grid_curriculum is not None
      and c.commands["twist"].grid_curriculum.unlock_min_vel_ratio is None,
    )
    check(
      f"{task} has no progress terms and stock startup friction DR",
      "velocity_progress_lin" not in c.rewards
      and "velocity_progress_ang" not in c.rewards
      and c.events["foot_friction"].mode == "startup"
      and c.events["foot_friction"].func is mjlab_dr.geom_friction
      and "severity_by_terrain_level" not in c.events["foot_friction"].params
      and "benign_low" not in c.events["foot_friction"].params,
      f"foot_friction mode {c.events['foot_friction'].mode}, "
      f"rewards {sorted(c.rewards)}",
    )
    c.scene.num_envs = 8
    c.observations["actor"].enable_corruption = False
    e = ManagerBasedRlEnv(cfg=c, device=DEVICE, render_mode=None)
    e.reset()
    a = torch.zeros(8, e.action_manager.total_action_dim, device=DEVICE)
    ok = True
    for _ in range(5):
      a.uniform_(-1.0, 1.0)
      _, r, *_ = e.step(a)
      ok = ok and bool(torch.isfinite(r).all())
    check(f"{task} still builds and steps finite", ok)
    ft = e.command_manager.get_term("twist")
    check(
      f"{task} ratio machinery dormant (no accumulators allocated)",
      ft.grid_enabled
      and not ft.grid_ratio_enabled
      and not hasattr(ft, "grid_achieved_lin_sum"),
    )
    # Same synthetic IDLE episode the v21b ratio gate blocks (case (a) in
    # section 6b): with the knob None the pre-existing frac-only behavior
    # must be unchanged — the idle episode still unlocks the neighbor.
    frm = e.reward_manager
    fdt = e.step_dt if frm._scale_by_dt else 1.0
    fw_lin = frm.get_term_cfg("track_linear_velocity").weight
    fw_ang = frm.get_term_cfg("track_angular_velocity").weight
    fsteps = 200
    fcell = 13 * ft._grid_n_wz + 4
    e.episode_length_buf[0] = fsteps
    frm._episode_sums["track_linear_velocity"][0] = 0.9 * fsteps * fdt * fw_lin
    frm._episode_sums["track_angular_velocity"][0] = 0.9 * fsteps * fdt * fw_ang
    ft.grid_cell_index[:] = -1
    ft.grid_cell_index[0] = fcell
    ft.vel_command_b[0, 0] = 0.225
    ft.vel_command_b[0, 1] = 0.0
    ft.vel_command_b[0, 2] = 0.05
    ft.grid_weights.copy_(ft.grid_seed_mask.float())
    fbefore = ft.grid_weights.clone()
    local_mdp.command_grid_adaptive(
      e, torch.arange(1, device=DEVICE), command_name="twist"
    )
    fexpected = fbefore.clone()
    fexpected[14, 4] = 0.2
    check(
      f"{task} frac-only unlock decision unchanged (idle episode unlocks)",
      bool(torch.allclose(ft.grid_weights, fexpected)),
      f"(14,4): {ft.grid_weights[14, 4].item():.2f} (want 0.20)",
    )
    e.close()
    del e

  rl = load_rl_cfg("XGOLite-V21B")
  check(
    "V21B PPO cfg: xgolite_v21b dir, 3000 iters, mirror loss/aug OFF",
    rl.experiment_name == "xgolite_v21b"
    and rl.max_iterations == 3000
    and rl.algorithm.symmetry_cfg is None,
    f"exp {rl.experiment_name}, iters {rl.max_iterations}",
  )

  print()
  if _failures:
    print(f"OVERALL: FAIL ({len(_failures)} failed: {', '.join(_failures)})")
    raise SystemExit(1)
  print("OVERALL: PASS")


if __name__ == "__main__":
  main()

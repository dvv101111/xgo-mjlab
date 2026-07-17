"""XGOLite-V21B: rough-terrain training on the v21a gait stack (v21 stage 2).

Derives from ``xgolite_v21a_env_cfg`` — the whole gait-family stack (ORC
phase-contact reward on the scheduled clock, walk member, symmetry reward,
widened vy) and the v20 measured plant carry over unchanged — and rebuilds
the terrain machinery the flat lineage stripped out
(env_cfgs.py:xgolite_flat_env_cfg deletes the generator terrain, the
terrain_scan sensor, both height_scan obs and the terrain curriculum; every
registered task derives from it, so the pieces are RECREATED here on top of
the v21a cfg rather than un-flattened). Design source:
docs/research/v21-terrain-gait-litreview-2026-07-15.md, shortlist A items
A2-A5 (A1, the estimator head, is deliberately OUT of scope for v21b).

1. TERRAIN (A3): scaled procedural families, 10 difficulty rows x 10 type
   columns, 2.0 m tiles (~the mid-size 8 m tile / 4; strict /5 geometry is
   applied to the FEATURES, the tile is kept slightly larger so a 20 s
   episode wanders across fewer difficulty boundaries). Feature scaling
   from the robot: thigh link 0.0549 m, calf hip-to-foot 0.0724 m
   (|fromto| of the calf geom), standing height 0.116 m — "leg length" for
   the lit review's <= 0.5-leg-length obstacle rule is the ~0.06 m link:
   - pyramid stairs +/- inverted, step height 0.005 -> 0.030 m by
     difficulty (0.030 = 0.5 x 0.06 m link = 26% of standing height; the
     A1-class evidence clears 0.5-0.75 leg lengths, we take the
     conservative end for 0.22 N*m servos), step run 0.08 m (/5 of the
     Isaac 0.3 m tread is 0.06; widened one foot-swing so the 2-6
     control-step transport delay lands touchdowns on the tread rather
     than the edge — Extreme Parkour's NoClear edge-landing failure);
   - random rough, height noise 3 -> 15 mm by difficulty (the lit /5
     household mapping: carpet pile, gravel-as-boulders), 5 cm cells
     (the strict-/5 2 cm grid overflows mjwarp's 50-triangle hfield
     collision budget under a fallen base box — see the sub-terrain cfg
     note; heights keep the lit mapping, the wavelength coarsens);
   - pyramid slopes +/- inverted, grade 0 -> 0.27 (~15 deg — the
     household-ramp ceiling from the lit mapping);
   - wave, amplitude 0 -> 0.03 m (same 0.5-link cap), wavelength 0.5 m;
   - flat, weight 0.2 (the spec anchor; also the grid-curriculum regime).
   Column weights are HIM-leaning (arXiv:2312.11460 makes stairs the
   dominant blind terrain): stairs family 0.4 = half of all non-flat mass.
   max_init_terrain_level=2 so early episodes spawn on rows 0-2 only.

2. BLIND actor (frozen 49-dim contract): the terrain_scan raycast
   (0.32 x 0.20 m grid, 0.02 m spacing = 187 rays, /5 of the mid-size
   1.6 x 1.0 / 0.1 pattern) and its height_scan observation are wired to
   the CRITIC only (asymmetric actor-critic, the lit's blind minimum);
   the actor keeps exactly the deployed obs frame. NOTE the honest
   expectation from the lit review negatives: plain H=5 history with no
   estimator head scored 20.5% rough-terrain survival in DreamWaQ's
   ablation — v21b's privileged critic + terrain curriculum will help,
   but blind-rough performance is expected to be the weak axis until a
   v21c estimator head lands.

3. Terrain-relative rewards: ``track_base_height`` and ``feet_clearance``
   assume a z=0 floor; V21B swaps them for the terrain-relative variants
   (``track_base_height_terrain``, ``feet_clearance_terrain``) that
   measure against the scan-derived local ground height (under the base /
   under each foot). Flat tasks keep the original functions — no behavior
   change outside V21B. Orientation terms (body_orientation_cmd_l2,
   body_ang_vel) stay world-frame flat-pull: on <= 15 deg ramps the
   gravity-aligned posture prior is acceptable (and desirable on steps);
   terrain-frame orientation is out of scope.

4. Compliance DR (A2): ``foot_contact_compliance`` randomizes the contact
   solref time constant up to 0.4 s (Singh arXiv:2504.13619, real
   foam/grass/mattress transfer) + solimp (d0, dmax, width) around the
   MuJoCo default, per env, written to the priority-1 foot pads
   (equivalent to softening the ground under that env — see the event
   docstring). Per-env writes are real: mjwarp carries
   geom_solref/geom_solimp per world once expanded. v21b-v5: the draw
   moved from startup to RESET mode with terrain-level severity coupling
   (point 10) — the ranges above are the top-row endpoints.

5. Friction (A4): per-env foot friction with a low tail down to 0.05 —
   the only range covering real ice (Rapid Locomotion arXiv:2205.02824)
   — plus ``foot_slip_event``: one random foot drops to U(0.02, 0.10)
   for U(0.2, 0.5) s every U(4, 8) s per env (Miki arXiv:2201.08117 slip
   injection). Both are effective against terrain geoms because the pads
   carry priority 1 (MuJoCo max-mixing gotcha). v21b-v5: both are
   severity-coupled to the terrain level (point 10).

6. Terrain curriculum (A5): ``terrain_levels_reward_gated`` — promotion
   per episode instead of distance (meaningless at 577 g); demotion only
   for envs that were actually commanded to move (the low-speed
   demotion-coupling bug guard, Isaac Lab #969/#1492/#1685). Composes
   with the v18range command grid: terrain moves env origins, the grid
   moves command-cell weights. v21b-v5: the promote/demote METRIC moved
   from tracking-reward fractions to the honest achieved-velocity ratio
   (point 11).

7. Spawning: reset scatter shrunk to +/-0.25 m (stays near the tile's
   spawn patch) and the reset z range raised to a (0.02, 0.04) m drop
   above the local terrain origin — clears the worst-case +6 mm local
   bump above a rough tile's mean-height origin with margin, settles in
   under 0.1 s.

8. Command-grid unlock ratio gate (v21b-v3, from the v21b-v2 10k
   postmortem — docs/research/v19-postmortem-v4-plant-2026-07-15.md,
   gate-metric fix): the frac-based grid gates are reward FRACTIONS, so
   idle episodes on lethal terrain pass them on cells nobody tracks (the
   v19-v5 idle-inversion; v2 unlocked 100% of cells by iter 2000 at
   seed_tracking 0.11). ``grid_curriculum.unlock_min_vel_ratio = 0.5``
   additionally requires a cell's episodes to achieve >= 50% of the
   commanded velocity (episode-mean, directional, per commanded axis)
   before the cell can unlock. Opt-in: every other task keeps ``None`` =
   bit-identical frac-only behavior.

9. Velocity-progress rewards (v21b-v5, from the v21b-v4 10k postmortem):
   three runs failed identically — ON THIS TERRAIN, STANDING STILL IS THE
   DOMINANT LOCAL OPTIMUM. The relative-sigma tracking family pays ~0.42
   raw at zero velocity over the seed command band (0.489 at cmd 0.08),
   ``stand_still`` only fires at |cmd| <= 0.1, so nothing priced idling
   under a move command; v4's warm walker abandoned locomotion in ~250
   iters and sat at vx_ach ~ 0.000 for the remaining 9750.
   ``velocity_progress_lin``/``velocity_progress_ang`` pay the achieved
   fraction of the commanded velocity (directional, clip [0, 1], zero
   when the axis is uncommanded — the grid ratio-gate math as a per-step
   reward). Weight sizing in the wiring comment below.

10. Hazard-DR severity scaled by terrain level (v21b-v5): the v4 autopsy's
   second root cause — compliance/slip/ice DR ran at FULL severity from
   iteration 0 on ALL terrain rows, so the flat-trained warm gait crashed
   immediately (episode length 19 at iter 0) and the gradient learned
   "walking = crashing" before terrain skill could form. All three hazard
   channels now interpolate severity per env with s = level / top_row
   (row 0: timeconst <= 0.05 s near-default contacts, friction low end
   0.4, slips are no-ops; top row: the full point-4/5 ranges — hazards
   arrive WITH terrain skill). Compliance + friction redraw on every
   RESET so severity tracks the env's current level; the slip drop is
   scaled at fire time. Opt-in per event term — other tasks unchanged.

11. Terrain promotion on the honest velocity ratio (v21b-v5.1, from the
   v5 live run killed at iter 1108): the progress rewards fixed the
   idling (velocity_progress_lin 1.26 -> 2.12, eplen 876, the policy
   genuinely walks), but terrain_levels collapsed 0.97 -> 0.02 by iter
   500 and pinned there (seed_tracking 0.26). Root cause: promotion
   gated on per-episode relative-sigma tracking FRACTIONS (>= 0.70/0.55)
   — a metric that scores STANDING at slow commands (~0.49 partial
   credit) HIGHER than honest ratio-0.5-0.6 walking (~0.26). v4's
   stander passed those gates in the tail (hollow promotions to 0.33);
   v5's honest walker never passed, and the frac demote bar (0.35 > its
   ~0.26) demoted every env to row 0 — freezing the point-10 hazard
   severity at s ~ 0, so no terrain skill could ever form. Fix:
   promotion requires the directional achieved/commanded velocity ratio
   >= 0.5 on EVERY commanded axis (the same honest metric as the grid
   unlock gate and the progress rewards, same 0.05/0.1 axis gates);
   demotion when ANY commanded axis drops below 0.25 (the frac demote is
   replaced, not kept — its bar sits above an honest walker's frac and
   would keep demoting exactly the policies this fixes). Episodes
   commanding neither axis keep the frac gates; the moving/min-episode
   guards are unchanged. Opt-in params, default off — other tasks
   bit-identical.
"""

import dataclasses

import mujoco
import numpy as np

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg, TerrainOutput

from src.tasks.velocity import mdp as local_mdp

from .env_cfgs import FOOT_PAD_GEOMS
from .v20 import V20_GAMMA_ANG, V20_GAMMA_LIN
from .v21a import xgolite_v21a_env_cfg

# --- Terrain geometry (see module docstring point 1 for the leg math). ---
V21B_TILE_SIZE = (2.0, 2.0)          # m; mid-size 8 m tile scaled ~/4
V21B_NUM_ROWS = 10                   # difficulty levels
V21B_NUM_COLS = 10                   # ignored by mjlab >= 1.5 curriculum mode
                                     # (one column per family; proportions
                                     # weight spawning only)
V21B_MAX_INIT_LEVEL = 2              # initial spawns on rows 0-2 only
V21B_STEP_HEIGHT_RANGE = (0.005, 0.030)  # m; max = 0.5 x 0.06 m leg link
V21B_STEP_WIDTH = 0.08               # m tread run
V21B_ROUGH_NOISE_RANGE = (0.003, 0.015)  # m; lit /5 mapping 3-15 mm
V21B_SLOPE_RANGE = (0.0, 0.27)       # grade; atan(0.27) ~ 15 deg
V21B_WAVE_AMPLITUDE_RANGE = (0.0, 0.03)  # m

# --- Critic terrain scan (/5 of the mid-size 1.6 x 1.0 / 0.1 grid). ---
V21B_SCAN_SIZE = (0.32, 0.20)
V21B_SCAN_RESOLUTION = 0.02          # -> 17 x 11 = 187 rays
V21B_SCAN_MAX_DISTANCE = 2.0

# --- Compliance DR (Singh arXiv:2504.13619; module docstring point 4). ---
# Full-severity (top terrain row) endpoints; see the severity block below.
V21B_SOLREF_TIMECONST_RANGE = (0.02, 0.4)
V21B_SOLIMP_D0_RANGE = (0.6, 0.9)
V21B_SOLIMP_DMAX_RANGE = (0.9, 0.97)
V21B_SOLIMP_WIDTH_RANGE = (0.001, 0.010)

# --- Friction + slip events (module docstring point 5). ---
# Full-severity (top terrain row) endpoints; see the severity block below.
V21B_FRICTION_RANGE = (0.05, 2.0)
V21B_SLIP_MU_RANGE = (0.02, 0.10)
V21B_SLIP_DURATION_RANGE = (0.2, 0.5)   # s
V21B_SLIP_INTERVAL_RANGE = (4.0, 8.0)   # s between slips, per env

# --- Hazard-DR severity coupling (module docstring point 10). ---
# Benign (terrain row 0, s=0) anchors: near-rigid contacts and no ice.
V21B_COMPLIANCE_BENIGN_TIMECONST = 0.05  # s; timeconst hi endpoint at s=0
V21B_FRICTION_BENIGN_LOW = 0.4           # friction low endpoint at s=0

# --- Velocity-progress rewards (module docstring point 9). ---
V21B_PROGRESS_LIN_WEIGHT = 6.0
V21B_PROGRESS_ANG_WEIGHT = 6.0

# --- Terrain curriculum gates (module docstring point 6). ---
V21B_TERRAIN_GAMMA_LIN = V20_GAMMA_LIN   # 0.70
V21B_TERRAIN_GAMMA_ANG = V20_GAMMA_ANG   # 0.55
# Achieved-velocity AND-gate for command-grid unlocks (module docstring
# point 8): a cell also needs mean achieved/commanded velocity ratio >= 0.5
# on every commanded axis. 0.5 sits well below a competent tracker (the
# frac gates 0.70/0.55 imply ratios ~0.8+) and well above the v21b-v2
# collapse (vx_ach 0.002 on all forward commands -> ratio ~0.01).
V21B_GRID_MIN_VEL_RATIO = 0.5
V21B_TERRAIN_DEMOTE_FRAC = 0.5           # demote below 0.5 * gamma_lin
V21B_TERRAIN_MIN_CMD_NORM = 0.1          # low-speed demotion guard
V21B_TERRAIN_MIN_EPISODE_FRAC = 0.5      # no promotion from early falls
# Terrain promotion/demotion on the honest achieved-velocity ratio (module
# docstring point 11; v5 postmortem). Promote = every commanded axis >= 0.5
# (the same honest-competence bar as the grid unlock ratio above); demote =
# any commanded axis < 0.25 (clearly failing, half the promote bar).
V21B_TERRAIN_PROMOTE_RATIO = 0.5
V21B_TERRAIN_DEMOTE_RATIO = 0.25


@dataclasses.dataclass(kw_only=True)
class HfDifficultyRandomUniformTerrainCfg(terrain_gen.HfRandomUniformTerrainCfg):
  """``HfRandomUniformTerrainCfg`` with difficulty-scaled noise amplitude.

  The upstream generator ignores ``difficulty`` for this family (matching
  Isaac Lab), which would make every row of the rough column equally hard
  and the terrain curriculum inert there. This variant interpolates the
  noise ceiling with difficulty: row 0 draws ~``noise_range[0]``-level
  bumps, the top row the full ``noise_range``. The floor is kept one
  ``noise_step`` above the minimum so the height lattice never collapses.
  """

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    lo, hi = self.noise_range
    eff_hi = max(lo + difficulty * (hi - lo), lo + self.noise_step)
    eff_cfg = dataclasses.replace(self, noise_range=(lo, eff_hi))
    return terrain_gen.HfRandomUniformTerrainCfg.function(
      eff_cfg, difficulty, spec, rng
    )


def xgolite_v21b_terrain_gen_cfg() -> TerrainGeneratorCfg:
  """Scaled-for-577g terrain grid (fresh cfg — ROUGH_TERRAINS_CFG untouched).

  mjlab >= 1.5 curriculum mode: ONE column per family in sub_terrains order
  (flat, rough, stairs, inv stairs, slope, inv slope, wave); the proportion
  fields weight robot spawning across columns, matching the old 10-column
  cumulative-proportion allocation in expectation.
  """
  return TerrainGeneratorCfg(
    size=V21B_TILE_SIZE,
    border_width=2.0,
    num_rows=V21B_NUM_ROWS,
    num_cols=V21B_NUM_COLS,
    curriculum=True,
    sub_terrains={
      "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.2),
      "random_rough": HfDifficultyRandomUniformTerrainCfg(
        proportion=0.1,
        noise_range=V21B_ROUGH_NOISE_RANGE,
        noise_step=0.003,
        # 0.05 m cells, not the strict /5 mapping's 0.02: mjwarp's hfield
        # collision budget is MJ_MAXCONPAIR=50 triangles per geom pair
        # (compile-time, collision_convex.py), and the 0.181 x 0.069 m
        # base box of a FALLEN robot (0.177 m square AABB when diagonal)
        # exceeds it below 0.05 m cells — measured per-column with random
        # actions: 0.02 -> ~54k, 0.03 -> ~7k, 0.04 -> ~3k overflow
        # warnings (= silently dropped contacts), 0.05 -> 0. Bump HEIGHTS
        # are unchanged (3-15 mm); only the bump wavelength coarsens
        # (5 cm ~ 0.83 leg links vs the proportional 0.33).
        horizontal_scale=0.05,
        vertical_scale=0.001,
        border_width=0.10,
      ),
      "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
        proportion=0.2,
        step_height_range=V21B_STEP_HEIGHT_RANGE,
        step_width=V21B_STEP_WIDTH,
        platform_width=0.6,
        border_width=0.2,
      ),
      "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
        proportion=0.2,
        step_height_range=V21B_STEP_HEIGHT_RANGE,
        step_width=V21B_STEP_WIDTH,
        platform_width=0.6,
        border_width=0.2,
      ),
      "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=0.1,
        slope_range=V21B_SLOPE_RANGE,
        platform_width=0.4,
        border_width=0.05,
        horizontal_scale=0.05,
        vertical_scale=0.001,
      ),
      "pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=0.1,
        slope_range=V21B_SLOPE_RANGE,
        platform_width=0.4,
        border_width=0.05,
        horizontal_scale=0.05,
        vertical_scale=0.001,
        inverted=True,
      ),
      "wave": terrain_gen.HfWaveTerrainCfg(
        proportion=0.1,
        amplitude_range=V21B_WAVE_AMPLITUDE_RANGE,
        num_waves=4,
        horizontal_scale=0.05,
        vertical_scale=0.001,
        border_width=0.05,
      ),
    },
    add_lights=True,
  )


def xgolite_v21b_ppo_runner_cfg(
  max_iterations: int = 3000,
) -> RslRlOnPolicyRunnerCfg:
  """v21a PPO cfg (mirror loss/aug OFF — the walk member breaks the trot
  mirror), fresh experiment dir, 3000-iter default (terrain from scratch
  converges slower than a flat polish).

  entropy_coef 0.001 (v21b-v3 postmortem): v21b is warm-started from a
  converged v21a actor whose loaded action std (~0.14) is already
  exploitation-annealed. The from-scratch coef (0.01) inflated std
  0.14->0.41 over 2k iters; on hazard-dense terrain that noise tripled
  the action-rate penalty, drove nonfoot-contact to -1.4/ep, tipped the
  per-step reward negative and made early termination reward-optimal
  (the v19-v1 suicide trap) — eplen 730->80 by iter 950. Fine-tuning
  regime: let the policy gradient move std, not a blanket entropy bonus.
  """
  from .aggressive import xgolite_aggressive_ppo_runner_cfg

  cfg = xgolite_aggressive_ppo_runner_cfg(
    "xgolite_v21b", max_iterations=max_iterations, symmetry=False
  )
  cfg.algorithm.entropy_coef = 0.001
  return cfg


def xgolite_v21b_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = xgolite_v21a_env_cfg(play=play)

  # 8. Achieved-velocity ratio AND-gate on the command-grid unlocks (the
  # v21b-v2 10k postmortem fix: idle episodes passed the frac-only gates on
  # cells nobody tracked and unlocked 100% of the grid by iter 2000 while
  # seed_tracking sat at 0.11). Opt-in knob — every other grid task keeps
  # unlock_min_vel_ratio=None and the pre-existing frac-only behavior.
  twist = cfg.commands["twist"]
  assert isinstance(twist, local_mdp.UniformVelocityCommandCfg)
  assert twist.grid_curriculum is not None
  twist.grid_curriculum.unlock_min_vel_ratio = V21B_GRID_MIN_VEL_RATIO

  # 1. Generator terrain (replaces the flat plane wholesale).
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=xgolite_v21b_terrain_gen_cfg(),
    max_init_terrain_level=V21B_MAX_INIT_LEVEL,
  )
  # Contact/solver headroom back at the rough-lineage values (the flat cfg
  # shrank them for the plane world): heightfield contacts + 187-ray scans.
  cfg.sim.nconmax = 35
  cfg.sim.njmax = 1500
  cfg.sim.mujoco.ccd_iterations = 100
  cfg.sim.contact_sensor_maxmatch = 128

  # 2. Critic-only terrain scan. The actor group is left untouched — the
  # deployed 49-dim frame is the whole point of "blind".
  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=V21B_SCAN_SIZE, resolution=V21B_SCAN_RESOLUTION),
    max_distance=V21B_SCAN_MAX_DISTANCE,
    exclude_parent_body=True,
    # Group 0 only = terrain geoms + foot pads. The default (0, 1, 2) lets
    # rays hit the robot's own group-1 VISUAL leg meshes (the base body is
    # excluded but leg bodies are not), which corrupts the ground-height
    # estimate under the feet by up to knee height. Foot-pad hits (also
    # group 0) are handled by the xy-exclusion in
    # ``terrain_height_under_feet``; for the scan-mean base height and the
    # critic obs, 0-2 pad hits among 187 rays are negligible.
    include_geom_groups=(0,),
    debug_vis=False,
  )
  # Base-body contact sensor for the terrain fall termination (below):
  # the flat task's thigh-spike termination is fatal-by-design on terrain
  # (spawn landings / step-edge brushes spike past 10 N for single
  # samples — the v21b v1 collapse), so terminal contact moves to the
  # BASE, sustained across the sensor history. Thigh/calf transients stay
  # penalized by the nonfoot_contact reward.
  base_ground = ContactSensorCfg(
    name="base_ground_touch",
    primary=ContactMatch(mode="body", pattern="base", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (terrain_scan, base_ground)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=local_mdp.sustained_illegal_contact,
    params={"sensor_name": base_ground.name, "force_threshold": 5.0},
  )
  cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": terrain_scan.name},
    scale=1.0 / V21B_SCAN_MAX_DISTANCE,
  )

  # 3. Terrain-relative height/clearance rewards (z=0 assumptions fixed for
  # V21B only; same weights/sigmas/targets as the flat terms they replace).
  tbh = cfg.rewards["track_base_height"]
  cfg.rewards["track_base_height"] = dataclasses.replace(
    tbh,
    func=local_mdp.track_base_height_terrain,
    params={**tbh.params, "sensor_name": terrain_scan.name},
  )
  fc = cfg.rewards["foot_clearance"]
  cfg.rewards["foot_clearance"] = dataclasses.replace(
    fc,
    func=local_mdp.feet_clearance_terrain,
    params={**fc.params, "sensor_name": terrain_scan.name},
  )

  # 9. Directional velocity-progress rewards (anti-idle; module docstring
  # point 9): pay the achieved fraction of the commanded velocity along the
  # command, clip [0, 1] — standing and reverse motion score 0, overshoot
  # saturates. Zero when the axis is uncommanded (lin gate |cmd_xy| >= 0.05,
  # ang gate |cmd_wz| >= 0.1 — the grid ratio-gate constants).
  #
  # WEIGHT SIZING (quantitative): mjlab's reward manager applies
  # contribution = raw * weight * step_dt (scale_by_dt default; step_dt =
  # decimation 10 x timestep 0.002 = 0.02 s — same convention as every
  # weight in this chain). v21b-v4's measured STANDING income was
  # ~ +0.03/step total, and standing's progress contribution is 0. Weight
  # 6.0 makes walking at ratio 0.5 add 6.0 * 0.5 * 0.02 = +0.06/step —
  # 2x the entire standing income on top of tracking partial credit that
  # already >= standing's (at ratio 0.5 the tracking error is halved) —
  # and +0.12/step (4x) at ratio 1.0. The ang term carries the same weight
  # so pure-rotation episodes (where the lin term is gated off) get the
  # same anti-idle pressure: +0.06/step at wz ratio 0.5, +0.12 at 1.0.
  cfg.rewards["velocity_progress_lin"] = RewardTermCfg(
    func=local_mdp.velocity_progress_lin,
    weight=V21B_PROGRESS_LIN_WEIGHT,
    params={"command_name": "twist"},
  )
  cfg.rewards["velocity_progress_ang"] = RewardTermCfg(
    func=local_mdp.velocity_progress_ang,
    weight=V21B_PROGRESS_ANG_WEIGHT,
    params={"command_name": "twist"},
  )

  # 4. Compliance DR on the priority-1 foot pads. v21b-v5: mode="reset" +
  # severity_by_terrain_level (module docstring point 10) — redrawn on
  # every reset at s = level/top_row, so row-0 envs get near-default
  # contacts (timeconst <= 0.05 s) and top-row envs the full Singh band.
  cfg.events["foot_compliance"] = EventTermCfg(
    func=local_mdp.foot_contact_compliance,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_PAD_GEOMS),
      "timeconst_range": V21B_SOLREF_TIMECONST_RANGE,
      "solimp_d0_range": V21B_SOLIMP_D0_RANGE,
      "solimp_dmax_range": V21B_SOLIMP_DMAX_RANGE,
      "solimp_width_range": V21B_SOLIMP_WIDTH_RANGE,
      "severity_by_terrain_level": True,
      "benign_timeconst_max": V21B_COMPLIANCE_BENIGN_TIMECONST,
    },
  )

  # 5. Friction low tail + transient per-foot slip events, severity-coupled
  # to the terrain level (module docstring point 10). The friction draw
  # moves from the startup dr.geom_friction to the reset-mode local event
  # whose LOW endpoint interpolates 0.4 (row 0: no ice under beginners) ->
  # 0.05 (top row: the Rapid-Locomotion ice tail); slip drops interpolate
  # from no-op (row 0) to the full Miki drop (top row) at fire time.
  # restore_on_reset=False on the slip term: the reset-mode friction event
  # redraws the base friction BEFORE the slip term's reset hook runs, so
  # the stale restore would overwrite the fresh draw (see the docstrings).
  cfg.events["foot_friction"] = EventTermCfg(
    func=local_mdp.foot_friction_terrain_scaled,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_PAD_GEOMS),
      "friction_range": V21B_FRICTION_RANGE,
      "benign_low": V21B_FRICTION_BENIGN_LOW,
    },
  )
  cfg.events["foot_slip"] = EventTermCfg(
    func=local_mdp.foot_slip_event,
    mode="step",  # step-mode state machine; see the event docstring.
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_PAD_GEOMS),
      "interval_range_s": V21B_SLIP_INTERVAL_RANGE,
      "duration_range_s": V21B_SLIP_DURATION_RANGE,
      "mu_range": V21B_SLIP_MU_RANGE,
      "severity_by_terrain_level": True,
      "restore_on_reset": False,
    },
  )

  # 7. Terrain-aware spawning: small drop above the local origin, scatter
  # kept near the tile's spawn patch.
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.25, 0.25),
    "y": (-0.25, 0.25),
    "z": (0.02, 0.04),
    "yaw": (-3.14, 3.14),
  }

  # 6. Reward-gated terrain curriculum (train cfg only; play keeps the
  # empty curriculum + randomize_terrain reset event from the play branch).
  if not play:
    # v21b-v5 (module docstring point 11): promotion/demotion gate on the
    # honest achieved-velocity ratio, not the relative-sigma reward fracs
    # (the frac gates remain for episodes commanding neither twist axis).
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=local_mdp.terrain_levels_reward_gated,
      params={
        "command_name": "twist",
        "lin_reward_name": "track_linear_velocity",
        "ang_reward_name": "track_angular_velocity",
        "gamma_lin": V21B_TERRAIN_GAMMA_LIN,
        "gamma_ang": V21B_TERRAIN_GAMMA_ANG,
        "demote_frac": V21B_TERRAIN_DEMOTE_FRAC,
        "min_cmd_norm": V21B_TERRAIN_MIN_CMD_NORM,
        "min_episode_frac": V21B_TERRAIN_MIN_EPISODE_FRAC,
        "promote_min_vel_ratio": V21B_TERRAIN_PROMOTE_RATIO,
        "demote_below_vel_ratio": V21B_TERRAIN_DEMOTE_RATIO,
      },
    )

  return cfg

"""
PiperD435iInterface
===================
Real-robot interface for 松灵 PIPER 6-DOF arm + Intel D435i depth camera.

Implements the same VectorEnv-like API used in fag_eval.py so it can be
used as a drop-in replacement for the LIBERO simulation environment.

Action format:
  Normalised mode (default, step()):
    action[0:3]  — delta XYZ in normalised units  (scaled to mm by XYZ_SCALE_MM)
    action[3:6]  — delta RPY in normalised units  (scaled to degrees by RPY_SCALE_DEG)
    action[6]    — gripper continuous [−1=open … +1=close]

  Physical mode (step_physical()):
    action[0:3]  — delta XYZ in metres  (clamped to ±MAX_DELTA_M)
    action[3:6]  — delta orientation, axis-angle or Euler in radians
    action[6]    — gripper in [-1, +1]  (−1=open/70mm, +1=closed/0mm)

Observation state (8-D, LIBERO-compatible):
    [x_m, y_m, z_m,  ax_rad, ay_rad, az_rad,  finger1_m, finger2_m]
    Position in metres; orientation as axis-angle (scipy as_rotvec);
    finger positions approximate Franka Panda convention:
        finger1_m =  (gripper_mm / GRIPPER_MAX_MM) * FINGER_MAX_M
        finger2_m = −finger1_m

Coordinate frame: PIPER base frame (X forward, Y left, Z up).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scale factors: pi0.5 normalised action → physical units
# ---------------------------------------------------------------------------
# Conservative values; tune on real hardware.
XYZ_SCALE_MM: float = 5.0         # mm per normalised unit (±1 → ±5 mm)
RPY_SCALE_DEG: float = 2.0        # degrees per normalised unit
GRIPPER_MAX_MM: float = 70.0      # PIPER gripper max opening (mm)
GRIPPER_EFFORT: int = 1000        # gripper closing force (0–1000)  ← was 500
COMMAND_HZ: float = 10.0          # Hz at which commands are sent

# Safety workspace limits — 50×50×50 cm around home [50, 0, 280] mm
# X: 50 ± 250 mm  |  Y: 0 ± 250 mm  |  Z: 280 ± 250 mm (floor protection at 80 mm)
WS_X_RANGE: Tuple[float, float] = (-200.0, 300.0)
WS_Y_RANGE: Tuple[float, float] = (-250.0, 250.0)
WS_Z_RANGE: Tuple[float, float] = ( 80.0,  530.0)

# Physical mode constants
# speed_pct=50 + 0.1s window → arm can execute ~30mm/step at reasonable speed
MAX_DELTA_XY_M: float  = 0.010    # XY: 10mm/step
MAX_DELTA_Z_M: float   = 0.030    # Z:  30mm/step (descent toward object)
MAX_DELTA_M: float     = MAX_DELTA_Z_M
MAX_DELTA_RAD: float   = 0.008    # rotation: ~0.46°/step (near-lock)
GRIPPER_EMA_ALPHA: float = 0.20   # EMA gripper smoothing
FINGER_MAX_M: float = 0.035       # Franka-like finger max travel (m) for state encoding
#   PIPER 70 mm total opening  →  35 mm per finger side  →  0.035 m

# ---------------------------------------------------------------------------
# LIBERO distribution alignment offsets
# ---------------------------------------------------------------------------
# pi0.5 was trained on LIBERO-Object (Franka Panda in MuJoCo).
# PIPER's workspace and orientation convention differ from LIBERO.
# These offsets shift PIPER state into LIBERO's training distribution so
# the policy receives in-distribution observations.
#
# Analysis (from libero10_normalization_stats.json):
#   LIBERO state mean  = [-0.042, 0.033, 0.841 m | 2.885, -0.672, -0.196 rad | ±0.028 m]
#   PIPER home state   = [+0.050, 0.000, 0.280 m | 0.000, +1.484,  0.000 rad | ±0.000 m]
#   Normalised delta Z = -2.18 std  ← most critical spatial axis
#   Normalised ax_rad  = -8.05 std  ← most critical orientation axis
#
# Strategy: apply offset so PIPER home looks like LIBERO home to the policy.
#   state_reported = state_piper + offset   (position, additive)
#   orient_reported = R_offset ∘ orient_piper  (rotation, compositional)
#
# PIPER home pose used as reference: [50mm, 0, 280mm, 0°, 85°, 0°]
_PIPER_HOME_POS_M  = np.array([ 0.0500,  0.0000,  0.2800])
_LIBERO_MEAN_POS_M = np.array([-0.0416,  0.0326,  0.8413])
LIBERO_POS_OFFSET_M: np.ndarray = _LIBERO_MEAN_POS_M - _PIPER_HOME_POS_M
# = [-0.0916, +0.0326, +0.5613]  (applied additively to EEF position in metres)

# Orientation: compositional offset R_off = R_libero_mean ∘ R_piper_home⁻¹
# Pre-computed at module load (scipy not available until first import)
_LIBERO_MEAN_AA    = np.array([ 2.8851, -0.6716, -0.1957])   # axis-angle (rad)
_PIPER_HOME_AA     = np.array([ 0.0000,  1.4835,  0.0000])   # axis-angle of [0°,85°,0°]

# Gripper finger offset: LIBERO demos have grippers mostly slightly open
# (mean ≈ ±0.028 m per finger); adding this offset prevents gripper being OOD
LIBERO_GRIP_OFFSET_M: float = 0.028  # added to finger1_m, subtracted from finger2_m


# ---------------------------------------------------------------------------
# Observation container
# ---------------------------------------------------------------------------

@dataclass
class RobotObs:
    """Structured observation dict compatible with pi0.5 preprocessor."""
    rgb_image: np.ndarray              # (H, W, 3) uint8, 640×480 by default
    depth_image: np.ndarray            # (H, W) uint16, mm units
    joint_positions: np.ndarray        # (6,) float32, degrees
    gripper_opening: float             # mm
    end_effector_pose: np.ndarray      # (6,) float32  [x,y,z mm, rx,ry,rz deg]
    task_description: str = ""

    def to_policy_dict(self, img_size: int = 224) -> Dict:
        """Convert to the observation dict format expected by pi0.5."""
        import cv2  # type: ignore
        rgb_resized = cv2.resize(self.rgb_image, (img_size, img_size))
        state = np.concatenate([
            self.joint_positions / 180.0 * np.pi,  # to radians
            [self.gripper_opening / GRIPPER_MAX_MM],
        ]).astype(np.float32)
        return {
            "pixels": {
                "image":  rgb_resized[None],    # (1, H, W, 3)
                "image2": rgb_resized[None],    # placeholder for wrist cam
            },
            "agent_pos": state[None],           # (1, 7)
            "task": [self.task_description],
        }


# ---------------------------------------------------------------------------
# D435i camera wrapper
# ---------------------------------------------------------------------------

class D435iCamera:
    """Intel RealSense D435i — RGB + depth capture."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width  = width
        self.height = height
        self.fps    = fps
        self._pipe  = None
        self._align = None

    def start(self):
        try:
            import pyrealsense2 as rs  # type: ignore
            # Hardware-reset first to recover from any prior unclean shutdown
            _ctx = rs.context()
            if _ctx.devices:
                _ctx.devices[0].hardware_reset()
                import time as _t; _t.sleep(2.5)
            self._pipe   = rs.pipeline()
            config       = rs.config()
            config.enable_stream(rs.stream.color, self.width, self.height,
                                 rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height,
                                 rs.format.z16, self.fps)
            self._pipe.start(config)
            self._align = rs.align(rs.stream.color)
            # Warm-up: discard first few frames (extended timeout for post-reset)
            for _ in range(5):
                self._pipe.wait_for_frames(timeout_ms=8000)
            logger.info("D435i camera started (%dx%d @ %d fps)",
                        self.width, self.height, self.fps)
        except ImportError:
            raise RuntimeError(
                "pyrealsense2 not installed. Run: "
                "uv pip install --python <venv> pyrealsense2"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start D435i camera: {e}")

    def flush(self, n_frames: int = 5):
        """Discard stale frames to reset internal pipeline state."""
        for _ in range(n_frames):
            try:
                self._pipe.wait_for_frames(timeout_ms=3000)
            except Exception:
                break

    def capture(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (rgb_bgr, depth_mm) as numpy arrays. Retries up to 3× on error."""
        import numpy as np
        last_err = None
        for attempt in range(3):
            try:
                frames      = self._pipe.wait_for_frames(timeout_ms=5000)
                aligned     = self._align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    raise RuntimeError("D435i: empty frame")
                rgb   = np.asanyarray(color_frame.get_data())   # BGR, uint8
                depth = np.asanyarray(depth_frame.get_data())   # uint16, mm
                return rgb, depth
            except Exception as e:
                last_err = e
                if attempt < 2:
                    logger.warning("Camera capture failed (attempt %d): %s — retrying", attempt + 1, e)
                    time.sleep(0.15)
        raise RuntimeError(f"D435i: capture failed after 3 attempts: {last_err}")

    def stop(self):
        if self._pipe is not None:
            self._pipe.stop()
            self._pipe = None
            logger.info("D435i camera stopped")


# ---------------------------------------------------------------------------
# PIPER6DOF arm wrapper
# ---------------------------------------------------------------------------

# Hz at which MotionCtrl_2 + EndPoseCtrl must be sent continuously
_CMD_HZ: float = 200.0
_CMD_PERIOD: float = 1.0 / _CMD_HZ   # 0.005 s


class PiperArm:
    """
    松灵 PIPER 6-DOF arm wrapper using piper_sdk V2.

    PIPER V2 requires commands to be sent at ~200 Hz continuously.
    All positions/angles in SDK units:
        position  →  0.001 mm   (so 100 mm = 100_000 units)
        angle     →  0.001 deg  (so  90 deg =  90_000 units)
        gripper   →  0.001 mm   (so  70 mm  =  70_000 units)
    """

    UNIT_POS = 1_000    # 1 mm = 1000 units
    UNIT_ANG = 1_000    # 1 deg = 1000 units
    UNIT_GRP = 1_000    # 1 mm = 1000 units

    def __init__(self, can_port: str = "can0"):
        self.can_port    = can_port
        self._arm        = None
        self._connected  = False

    def connect(self):
        try:
            from piper_sdk import C_PiperInterface_V2  # type: ignore
            self._arm = C_PiperInterface_V2(self.can_port)
            self._arm.ConnectPort()
            # Wait until EnablePiper succeeds (official SDK pattern)
            deadline = time.time() + 5.0
            while not self._arm.EnablePiper():
                time.sleep(0.01)
                if time.time() > deadline:
                    raise RuntimeError("EnablePiper timed out after 5 s")
            time.sleep(0.3)
            # Exit teaching/drag mode if active, then enter CAN EndPose mode
            self._arm.MotionCtrl_1(emergency_stop=0, track_ctrl=0, grag_teach_ctrl=0)
            time.sleep(0.2)
            self._arm.MotionCtrl_2(0x01, 0x00, 100, 0x00)  # CAN ctrl, MOVE_J, 100% spd
            time.sleep(0.2)
            # ---- Gripper initialisation (P0 fix) -------------------------
            # 1. Set gripper max-range parameter (must be done once per arm,
            #    stored in firmware; harmless to call again).
            #    Without this, gripper gives no feedback and won't move.
            #    See: demo/V2/V2_piper_set_gripper_param.py
            self._arm.GripperTeachingPendantParamConfig(100, 70, 1)
            time.sleep(0.3)
            # 2. Query back the gripper param (triggers firmware to load config)
            self._arm.ArmParamEnquiryAndConfig(4)
            time.sleep(0.2)
            # 3. SDK reset + enable sequence (from demo/V2/piper_ctrl_gripper.py)
            self._arm.GripperCtrl(0, GRIPPER_EFFORT, 0x02, 0)  # disable + clear error
            time.sleep(0.15)
            self._arm.GripperCtrl(0, GRIPPER_EFFORT, 0x01, 0)  # enable
            time.sleep(0.15)
            self._connected = True
            logger.info("PIPER arm connected on %s", self.can_port)
        except ImportError:
            raise RuntimeError(
                "piper_sdk not installed. Run: "
                "uv pip install --python <venv> piper_sdk"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect PIPER arm: {e}")

    def disconnect(self):
        if self._arm and self._connected:
            try:
                self._arm.DisablePiper()
                self._arm.DisconnectPort()
            except Exception as e:
                logger.warning("Error during arm disconnect: %s", e)
            self._connected = False
            logger.info("PIPER arm disconnected")

    # ------------------------------------------------------------------
    # State reading
    # ------------------------------------------------------------------

    def get_joint_positions_deg(self) -> np.ndarray:
        """Return 6 joint angles in degrees."""
        j = self._arm.GetArmJointMsgs().joint_state
        return np.array([
            j.joint_1, j.joint_2, j.joint_3,
            j.joint_4, j.joint_5, j.joint_6,
        ], dtype=np.float32) * 1e-3   # units → degrees

    def get_end_effector_pose(self) -> np.ndarray:
        """Return end-effector pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]."""
        e = self._arm.GetArmEndPoseMsgs().end_pose
        return np.array([
            e.X_axis  * 1e-3,   # mm
            e.Y_axis  * 1e-3,
            e.Z_axis  * 1e-3,
            e.RX_axis * 1e-3,   # degrees
            e.RY_axis * 1e-3,
            e.RZ_axis * 1e-3,
        ], dtype=np.float32)

    def get_gripper_mm(self) -> float:
        """Return gripper opening in mm (SDK unit: 0.001 mm → divide by 1000)."""
        msg = self._arm.GetArmGripperMsgs()
        # Field: gripper_state.grippers_angle (int, unit 0.001 mm)
        raw = getattr(msg.gripper_state, "grippers_angle", 0)
        return float(raw) * 1e-3  # → mm

    # ------------------------------------------------------------------
    # Command sending  (must be called at 200 Hz; send_for_seconds wraps this)
    # ------------------------------------------------------------------

    def _send_end_pose_once(self, x_u: int, y_u: int, z_u: int,
                             rx_u: int, ry_u: int, rz_u: int,
                             gripper_u: int, speed_pct: int = 2):
        """Send one EndPoseCtrl + GripperCtrl frame (call at 200 Hz)."""
        self._arm.MotionCtrl_2(0x01, 0x00, speed_pct, 0x00)
        self._arm.EndPoseCtrl(x_u, y_u, z_u, rx_u, ry_u, rz_u)
        self._arm.GripperCtrl(gripper_u, GRIPPER_EFFORT, 0x01, 0)

    def move_end_effector(
        self,
        target_pose_mm_deg: np.ndarray,
        gripper_mm: float = GRIPPER_MAX_MM,
        duration_s: float = 1.0 / COMMAND_HZ,
        speed_pct: int = 2,
    ):
        """
        Send end-effector target continuously for `duration_s` seconds at 200 Hz.

        target_pose_mm_deg: [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        gripper_mm: gripper opening in mm (0=closed, 70=open)
        duration_s: how long to stream the command (default = 1 control step)
        """
        x, y, z, rx, ry, rz = target_pose_mm_deg.tolist()
        x  = float(np.clip(x,  *WS_X_RANGE))
        y  = float(np.clip(y,  *WS_Y_RANGE))
        z  = float(np.clip(z,  *WS_Z_RANGE))
        gripper_mm = float(np.clip(gripper_mm, 0.0, GRIPPER_MAX_MM))

        x_u  = int(x  * self.UNIT_POS)
        y_u  = int(y  * self.UNIT_POS)
        z_u  = int(z  * self.UNIT_POS)
        rx_u = int(rx * self.UNIT_ANG)
        ry_u = int(ry * self.UNIT_ANG)
        rz_u = int(rz * self.UNIT_ANG)
        grp_u = int(gripper_mm * self.UNIT_GRP)

        end_t = time.time() + duration_s
        while time.time() < end_t:
            t0 = time.time()
            self._send_end_pose_once(x_u, y_u, z_u, rx_u, ry_u, rz_u, grp_u,
                                     speed_pct=speed_pct)
            elapsed = time.time() - t0
            sleep_t = _CMD_PERIOD - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def set_gripper(self, opening_mm: float, effort: int = GRIPPER_EFFORT):
        """Control gripper opening (0=fully closed, 70=fully open in mm)."""
        opening_mm = float(np.clip(opening_mm, 0.0, GRIPPER_MAX_MM))
        self._arm.GripperCtrl(
            gripper_angle  = int(opening_mm * self.UNIT_GRP),
            gripper_effort = effort,
            gripper_code   = 0x01,
            set_zero       = 0,
        )

    def go_home(self):
        """
        Move arm to safe home pose using EndPoseCtrl (streams at 200 Hz).
        Home pose: X=50mm, Y=0, Z=280mm (well above table), RX=0, RY=85deg, RZ=0.

        speed_pct=5 / duration_s=45: covers full workspace diagonal (~600mm) at
        ~10mm/s comfortable speed, and always reaches home from any starting point.
        (speed_pct=1 only travels ~12mm in 6s — insufficient if far from home.)
        """
        home = np.array([50.0, 0.0, 280.0, 0.0, 85.0, 0.0], dtype=np.float32)
        self.move_end_effector(home, gripper_mm=GRIPPER_MAX_MM,
                               duration_s=45.0, speed_pct=5)


# ---------------------------------------------------------------------------
# Combined interface (VectorEnv-compatible)
# ---------------------------------------------------------------------------

class PiperD435iInterface:
    """
    Combined PIPER6DOF arm + D435i camera interface.

    Implements VectorEnv-like API so it can replace the LIBERO simulation
    in fag_eval.py without changing the evaluation loop logic.

    Parameters
    ----------
    task_name:  natural-language task description
    can_port:   CAN interface name (default "can0")
    max_steps:  max steps per episode before forced termination
    img_size:   resize captured images to this square size
    """

    def __init__(
        self,
        task_name: str,
        can_port: str = "can0",
        max_steps: int = 200,
        img_size:  int = 224,
    ):
        self.task_name  = task_name
        self.can_port   = can_port
        self.max_steps  = max_steps
        self.img_size   = img_size
        self.num_envs   = 1

        self._cam = D435iCamera()
        self._arm = PiperArm(can_port=can_port)
        self._step_count  = 0
        self._prev_eef    = np.zeros(6, dtype=np.float32)
        self._gripper_ema = GRIPPER_MAX_MM  # EMA state, starts fully open
        self._connected   = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self):
        """Connect hardware; must be called before reset/step."""
        logger.info("Connecting PIPER arm + D435i camera …")
        self._arm.connect()
        self._cam.start()
        self._arm.go_home()
        self._connected = True
        logger.info("Hardware connected and at home position")

    def disconnect(self):
        """Safely park arm and release camera."""
        if self._connected:
            try:
                self._arm.go_home()
            except Exception:
                pass
            self._cam.stop()
            self._arm.disconnect()
            self._connected = False

    # ------------------------------------------------------------------
    # VectorEnv-compatible API
    # ------------------------------------------------------------------

    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """
        Reset the robot to home position and return initial observation.
        Returns (obs_dict, info_dict) matching LIBERO env format.
        """
        if not self._connected:
            raise RuntimeError("Call connect() before reset()")

        logger.info("Resetting robot to home position …")
        self._arm.go_home()
        self._step_count  = 0
        self._gripper_ema = GRIPPER_MAX_MM  # reset gripper EMA to open
        self._prev_eef    = self._arm.get_end_effector_pose()

        # Flush stale frames accumulated during go_home (camera was idle ~45s)
        self._cam.flush(n_frames=8)
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray, bool, Dict]:
        """
        Execute one action and return (obs, reward, done, truncated, info).

        action: (1, 7) or (7,) array
            [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
            All in normalised [-1, 1] (xyz) or [-1, 1] (rpy) and [0, 1] (gripper)
        """
        if not self._connected:
            raise RuntimeError("Call connect() before step()")

        action = np.asarray(action).flatten()[:7]

        # Convert delta action to absolute target pose
        current_eef = self._arm.get_end_effector_pose()
        delta_xyz    = action[0:3] * XYZ_SCALE_MM
        delta_rpy    = action[3:6] * RPY_SCALE_DEG
        gripper_cmd  = float(action[6])

        target_eef = current_eef.copy()
        target_eef[0:3] += delta_xyz
        target_eef[3:6] += delta_rpy
        # gripper_cmd is in [-1, +1]:  -1 = open (70mm),  +1 = closed (0mm)
        gripper_opening_mm = (1.0 - np.clip(gripper_cmd, 0.0, 1.0)) * GRIPPER_MAX_MM

        # Stream command at 200 Hz for one control period (replaces single send + sleep)
        self._arm.move_end_effector(
            target_eef,
            gripper_mm=gripper_opening_mm,
            duration_s=1.0 / COMMAND_HZ,
            speed_pct=8,
        )

        self._step_count += 1
        self._prev_eef = self._arm.get_end_effector_pose()

        obs     = self._get_obs()
        reward  = np.array([0.0])
        done    = np.array([self._step_count >= self.max_steps])
        info: Dict = {"success": [False]}   # success determined externally

        return obs, reward, done, False, info

    def step_physical(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict, np.ndarray, np.ndarray, bool, Dict]:
        """
        Execute one action given in PHYSICAL units (metres, radians, gripper norm).

        Use this when the action comes from the pi0.5 postprocessor (un-normalised
        LIBERO-format action), NOT from the normalised action space.

        action: (1, 7) or (7,) array
          [δx_m, δy_m, δz_m,  δrx_rad, δry_rad, δrz_rad,  gripper_cmd]
          δXYZ clamped to ±MAX_DELTA_M (0.05 m = 50 mm)
          δRPY clamped to ±MAX_DELTA_RAD (0.20 rad ≈ 11.5°)
          gripper_cmd: −1 = fully open (70 mm), +1 = fully closed (0 mm)
        """
        if not self._connected:
            raise RuntimeError("Call connect() before step_physical()")

        action = np.asarray(action).flatten()[:7]

        # Asymmetric per-axis safety clamp
        # XY: tightly limited to suppress lateral drift (object is directly below EEF)
        # Z:  larger limit to allow descent toward object
        delta_xy_m    = np.clip(action[0:2], -MAX_DELTA_XY_M, MAX_DELTA_XY_M)
        delta_z_m     = np.clip(action[2:3], -MAX_DELTA_Z_M,  MAX_DELTA_Z_M)
        delta_xyz_m   = np.concatenate([delta_xy_m, delta_z_m])
        delta_rpy_rad = np.clip(action[3:6], -MAX_DELTA_RAD,  MAX_DELTA_RAD)
        gripper_cmd   = float(action[6])

        # Convert to PIPER units
        current_eef = self._arm.get_end_effector_pose()   # mm, degrees
        target_eef  = current_eef.copy()
        target_eef[0:3] += delta_xyz_m * 1000.0           # m → mm
        target_eef[3:6] += np.degrees(delta_rpy_rad)      # rad → degrees

        # gripper_cmd: −1=open, +1=close  →  mm opening (−1→70, +1→0)
        gripper_raw = (1.0 - np.clip(gripper_cmd, -1.0, 1.0)) / 2.0 * GRIPPER_MAX_MM
        # EMA smoothing: damps high-frequency oscillation caused by OOD policy outputs
        self._gripper_ema = (GRIPPER_EMA_ALPHA * gripper_raw
                             + (1.0 - GRIPPER_EMA_ALPHA) * self._gripper_ema)
        gripper_mm = float(self._gripper_ema)

        self._arm.move_end_effector(
            target_eef,
            gripper_mm=gripper_mm,
            duration_s=1.0 / COMMAND_HZ,
            speed_pct=8,
        )

        self._step_count += 1
        self._prev_eef = self._arm.get_end_effector_pose()

        obs    = self._get_obs()
        reward = np.array([0.0])
        done   = np.array([self._step_count >= self.max_steps])
        info: Dict = {"success": [False]}

        return obs, reward, done, False, info

    # ------------------------------------------------------------------
    # VectorEnv attribute shims expected by lerobot utils
    # ------------------------------------------------------------------

    @property
    def envs(self) -> List["PiperD435iInterface"]:
        """Mimic gym.vector.VectorEnv.envs — returns [self]."""
        return [self]

    @property
    def task(self) -> str:
        """Task description used by add_envs_task."""
        return self.task_name

    @property
    def task_description(self) -> str:
        """Alternate task-description attribute used by some lerobot utils."""
        return self.task_name

    def call(self, method: str) -> List:
        """Compatibility shim for VectorEnv.call()."""
        if method == "_max_episode_steps":
            return [self.max_steps]
        if method in ("task", "task_description"):
            return [self.task_name]
        return [None]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict:
        """
        Capture current RGB-D frame and arm state.

        State vector (8-D, LIBERO-compatible format for pi0.5):
          [x_m, y_m, z_m,  ax_rad, ay_rad, az_rad,  finger1_m, finger2_m]

          - Position in metres (PIPER mm ÷ 1000)
          - Orientation as axis-angle vector (radians), via scipy Rotation
          - Two symmetric finger positions in metres (Franka Panda convention):
              finger1 = +(gripper_mm / GRIPPER_MAX_MM) * FINGER_MAX_M
              finger2 = −finger1

        Image: BGR (D435i) → RGB (pi0.5 expects RGB)
        """
        from scipy.spatial.transform import Rotation  # type: ignore

        bgr, depth = self._cam.capture()
        eef_pose   = self._arm.get_end_effector_pose()  # [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        gripper_mm = self._arm.get_gripper_mm()

        import cv2
        rgb         = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # BGR → RGB  (Bug-fix)
        rgb_resized = cv2.resize(rgb, (self.img_size, self.img_size))

        # ---- 8-D state: raw PIPER EEF pose (no LIBERO alignment offsets) ----
        # Sending honest PIPER state is better than fake LIBERO offsets, which
        # place the state exactly at LIBERO mean → policy outputs ~0 actions.
        pos_m      = eef_pose[:3] / 1000.0                           # mm → metres
        rpy_rad    = np.deg2rad(eef_pose[3:6])                       # deg → rad
        R_current  = Rotation.from_euler("xyz", rpy_rad)
        axis_angle = R_current.as_rotvec()                           # raw axis-angle

        # Franka-like symmetric finger values (no LIBERO offset)
        finger_m = (gripper_mm / GRIPPER_MAX_MM) * FINGER_MAX_M

        state = np.array([
            pos_m[0],      pos_m[1],      pos_m[2],
            axis_angle[0], axis_angle[1], axis_angle[2],
            finger_m, -finger_m,
        ], dtype=np.float32)

        return {
            "pixels": {
                "image":  rgb_resized[None],   # (1, H, W, 3) uint8 RGB
                "image2": rgb_resized[None],   # wrist placeholder
            },
            "agent_pos": state[None],          # (1, 8)
            "task": [self.task_name],
            # Extra fields for FAG analysis / diagnostics
            "_depth": depth,
            "_eef_pose": eef_pose,
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

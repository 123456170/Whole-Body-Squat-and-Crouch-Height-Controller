import streamlit as st
import numpy as np
import cv2
import math
import time
import os
import tempfile
from pathlib import Path

try:
    import pybullet
except Exception:
    pybullet = None

try:
    import mediapipe as mp
except Exception:
    mp = None

st.set_page_config(
    page_title="Whole-Body Squat Height Controller",
    page_icon="🤖",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------------
FRAME_W = 1280
FRAME_H = 520
LEFT_W = 470

FPS = 15
DEMO_DURATION = 60.0
MAX_VEL = 0.30

THIGH = 0.42
SHANK = 0.40
TORSO_LEN = 0.55
HEAD_LEN = 0.19

HEEL_X = -0.09
TOE_X = 0.20
SUPPORT_CENTER = (HEEL_X + TOE_X) / 2.0

STAND_HEIGHT = 0.815
MIN_HEIGHT = 0.42
MIN_SAFE_HEIGHT = 0.48

TORQUE_BUDGET = 150.0

COARSE_SAMPLES = 37
REFINE_SAMPLES = 13


# ----------------------------------------------------------------------------
# One-Euro filter for deliberate hand-height command smoothing
# ----------------------------------------------------------------------------
def smoothing_factor(te, cutoff):
    r = 2.0 * math.pi * cutoff * te
    return r / (r + 1.0)


class OneEuro:
    def __init__(self, mincutoff=0.55, beta=0.02, dcutoff=1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0

    def __call__(self, x, dt):
        if dt <= 0:
            dt = 1.0 / FPS

        if self.x_prev is None:
            self.x_prev = float(x)
            self.dx_prev = 0.0
            return float(x)

        dx = (float(x) - self.x_prev) / dt
        a_d = smoothing_factor(dt, self.dcutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(dt, cutoff)

        x_hat = a * float(x) + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def w2p(p, ox, oy, scale):
    return (int(ox + float(p[0]) * scale), int(oy - float(p[1]) * scale))


def put_text(frame, text, x, y, color=(230, 230, 230), scale=0.55, thickness=1):
    cv2.putText(
        frame,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_gauge(frame, x, y, w, h, pct, label):
    put_text(frame, label, x, y - 4, (200, 210, 220), 0.45, 1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (110, 115, 125), 1)

    pct = float(np.clip(pct, 0.0, 1.0))
    fill = int(w * pct)

    if pct < 0.70:
        color = (0, 200, 120)
    elif pct < 0.90:
        color = (0, 200, 255)
    else:
        color = (0, 0, 255)

    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)


# ----------------------------------------------------------------------------
# Robot model / IK / CoM centering
# ----------------------------------------------------------------------------
def emergency_pose():
    H = np.array([0.0, STAND_HEIGHT], dtype=float)
    K = np.array([0.04, 0.42], dtype=float)
    A = np.array([0.0, 0.0], dtype=float)

    torso_dir = np.array([math.sin(0.12), math.cos(0.12)], dtype=float)
    shoulder = H + TORSO_LEN * torso_dir
    head = shoulder + HEAD_LEN * np.array([math.sin(0.04), math.cos(0.04)], dtype=float)
    hand = np.array([0.30, 1.35], dtype=float)
    com = np.array([SUPPORT_CENTER, 0.80], dtype=float)

    return {
        "hip": H,
        "knee": K,
        "ankle": A,
        "heel": np.array([HEEL_X, 0.0], dtype=float),
        "toe": np.array([TOE_X, 0.0], dtype=float),
        "shoulder": shoulder,
        "head": head,
        "hand": hand,
        "com": com,
        "knee_flex": 0.15,
        "ankle_pitch": 0.10,
        "torso_pitch": 0.12,
        "height": STAND_HEIGHT,
        "knee_torque": 20.0,
        "feasible": True,
        "com_error": 0.0,
        "torque_pct": 20.0 / TORQUE_BUDGET,
        "message": "",
    }


def solve_pose(hx, hy, hand_norm):
    H = np.array([hx, hy], dtype=float)
    A = np.array([0.0, 0.0], dtype=float)

    d = float(np.linalg.norm(A - H))
    if d < abs(THIGH - SHANK) + 1e-4:
        return None
    if d > THIGH + SHANK + 1e-3:
        return None

    u = (A - H) / d
    a = (THIGH * THIGH - SHANK * SHANK + d * d) / (2.0 * d)
    h2 = THIGH * THIGH - a * a
    if h2 < 0.0:
        h2 = 0.0
    h = math.sqrt(h2)

    P = H + a * u
    candidates = []

    for sign in (1.0, -1.0):
        v = np.array([-u[1] * sign, u[0] * sign], dtype=float)
        K = P + h * v

        if K[1] < 0.03:
            continue

        if abs(float(np.linalg.norm(K - A)) - SHANK) > 0.01:
            continue

        ankle_pitch = math.atan2(K[0] - A[0], K[1] - A[1])
        if ankle_pitch < -0.35 or ankle_pitch > 0.80:
            continue

        v1 = H - K
        v2 = A - K
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))

        if n1 < 1e-6 or n2 < 1e-6:
            continue

        cosk = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        internal = math.acos(cosk)
        knee_flex = math.pi - internal

        if knee_flex < -0.08 or knee_flex > 2.35:
            continue

        torso_pitch = float(np.clip(0.10 + 0.45 * knee_flex, 0.08, 0.70))
        torso_dir = np.array([math.sin(torso_pitch), math.cos(torso_pitch)], dtype=float)

        shoulder = H + TORSO_LEN * torso_dir
        head_dir = np.array([math.sin(torso_pitch * 0.35), math.cos(torso_pitch * 0.35)], dtype=float)
        head = shoulder + HEAD_LEN * head_dir

        hn = float(np.clip(hand_norm, 0.0, 1.0))
        hand = np.array([0.30, 0.78 + 0.95 * hn], dtype=float)
        if hand[1] < shoulder[1]:
            hand[0] = 0.18

        foot_com = np.array([(HEEL_X + TOE_X) / 2.0, 0.04], dtype=float)
        shank_com = A + 0.43 * (K - A)
        thigh_com = H + 0.43 * (K - H)
        torso_com = H + 0.50 * TORSO_LEN * torso_dir
        head_com = shoulder + 0.50 * HEAD_LEN * head_dir
        arm_com = shoulder + 0.45 * (hand - shoulder)

        masses = np.array([1.2, 4.4, 10.2, 30.0, 5.2, 6.0], dtype=float)
        pts = np.vstack([foot_com, shank_com, thigh_com, torso_com, head_com, arm_com])
        total_mass = float(masses.sum())
        com = (pts * masses[:, None]).sum(axis=0) / total_mass

        knee_torque = total_mass * 9.81 * abs(float(com[0] - K[0])) + 9.0 * knee_flex
        feasible = bool(
            (knee_torque <= TORQUE_BUDGET * 1.02)
            and (abs(float(com[0] - SUPPORT_CENTER)) < 0.06)
        )

        candidates.append(
            {
                "hip": H.copy(),
                "knee": K.copy(),
                "ankle": A.copy(),
                "heel": np.array([HEEL_X, 0.0], dtype=float),
                "toe": np.array([TOE_X, 0.0], dtype=float),
                "shoulder": shoulder,
                "head": head,
                "hand": hand,
                "com": com,
                "knee_flex": knee_flex,
                "ankle_pitch": ankle_pitch,
                "torso_pitch": torso_pitch,
                "height": float(hy),
                "knee_torque": float(knee_torque),
                "feasible": feasible,
                "com_error": float(com[0] - SUPPORT_CENTER),
                "torque_pct": float(knee_torque / TORQUE_BUDGET),
                "message": "",
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda p: (
            not p["feasible"],
            -float(p["knee"][0]),
            abs(float(p["com"][0] - SUPPORT_CENTER)),
        )
    )

    return candidates[0]


def solve_squat(height, hand_norm):
    best = None
    best_score = 1e9
    best_hx = 0.0

    for hx in np.linspace(-0.30, 0.20, COARSE_SAMPLES):
        p = solve_pose(float(hx), float(height), hand_norm)
        if p is None:
            continue

        err = abs(float(p["com"][0] - SUPPORT_CENTER))
        over = max(0.0, float(p["knee_torque"]) - TORQUE_BUDGET)
        score = 1000.0 * over + 8.0 * err + 0.05 * max(0.0, p["knee_flex"] - 2.1)

        if score < best_score:
            best_score = score
            best = p
            best_hx = float(hx)

    if best is None:
        return None

    for hx in np.linspace(best_hx - 0.035, best_hx + 0.035, REFINE_SAMPLES):
        p = solve_pose(float(hx), float(height), hand_norm)
        if p is None:
            continue

        err = abs(float(p["com"][0] - SUPPORT_CENTER))
        over = max(0.0, float(p["knee_torque"]) - TORQUE_BUDGET)
        score = 1000.0 * over + 8.0 * err + 0.05 * max(0.0, p["knee_flex"] - 2.1)

        if score < best_score:
            best_score = score
            best = p
            best_hx = float(hx)

    best["com_error"] = float(best["com"][0] - SUPPORT_CENTER)
    best["torque_pct"] = float(best["knee_torque"] / TORQUE_BUDGET)
    return best


def pose_for_command(requested_height, hand_norm):
    h = float(np.clip(requested_height, MIN_SAFE_HEIGHT, STAND_HEIGHT))
    clamped = bool(requested_height < MIN_SAFE_HEIGHT - 1e-4)

    pose = solve_squat(h, hand_norm)

    for _ in range(6):
        if (
            pose is not None
            and pose["knee_torque"] <= TORQUE_BUDGET
            and abs(pose["com_error"]) <= 0.06
        ):
            break

        h = min(STAND_HEIGHT, h + 0.02)
        clamped = True
        pose = solve_squat(h, hand_norm)

    if pose is None:
        pose = solve_squat(STAND_HEIGHT, hand_norm)
        clamped = True

    if pose is None:
        pose = emergency_pose()
        clamped = True

    if clamped:
        pose["message"] = "Maximum safe depth reached"
    else:
        pose["message"] = ""

    return pose


def default_state():
    return {
        "filter": OneEuro(),
        "command_height": STAND_HEIGHT,
        "actual_height": STAND_HEIGHT,
        "target_height": STAND_HEIGHT,
        "hand_raw": 0.85,
        "hand_filtered": 0.85,
        "hand_norm": 0.85,
        "velocity_limited": False,
        "history": [],
        "last_pose": emergency_pose(),
        "message": "Demo ready",
    }


def step_state(state, hand_raw, dt):
    hand_raw = float(np.clip(hand_raw, -0.5, 1.2))
    hand_filtered = float(state["filter"](hand_raw, dt))
    hand_norm = float(np.clip(hand_filtered, 0.0, 1.0))

    target_height = MIN_HEIGHT + hand_norm * (STAND_HEIGHT - MIN_HEIGHT)

    prev_cmd = float(state["command_height"])
    max_delta = MAX_VEL * dt
    delta = target_height - prev_cmd
    velocity_limited = bool(abs(delta) > max_delta + 1e-6)

    cmd = prev_cmd + float(np.clip(delta, -max_delta, max_delta))
    state["command_height"] = cmd

    pose = pose_for_command(cmd, hand_norm)

    state["target_height"] = float(target_height)
    state["actual_height"] = float(pose["height"])
    state["hand_raw"] = hand_raw
    state["hand_filtered"] = hand_filtered
    state["hand_norm"] = hand_norm
    state["velocity_limited"] = velocity_limited
    state["last_pose"] = pose
    state["message"] = pose.get("message", "")

    state["history"].append(pose)
    if len(state["history"]) > 14:
        state["history"] = state["history"][-14:]

    return pose


# ----------------------------------------------------------------------------
# Demo trajectory matching the 60-second script
# ----------------------------------------------------------------------------
def demo_hand_height(t):
    if t < 12.0:
        # Stand -> crouch -> stand
        return 0.55 + 0.45 * math.cos(math.pi * t / 6.0)

    if t < 27.0:
        # Lower hand deliberately
        p = (t - 12.0) / 15.0
        return 1.0 - 0.85 * smoothstep(p)

    if t < 40.0:
        # Raise quickly to show velocity ramp limiting
        if t < 27.8:
            return 0.15 + 0.85 * smoothstep((t - 27.0) / 0.8)
        return 1.0

    if t < 52.0:
        # Push past depth limit
        if t < 40.6:
            return 1.0 - 1.30 * smoothstep((t - 40.0) / 0.6)
        return -0.30

    # Recover and show CoM centering near zero
    return -0.30 + 0.92 * smoothstep((t - 52.0) / 1.5)


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def draw_skeleton(frame, pose, ox, oy, scale, color, thickness=2):
    hip = pose["hip"]
    knee = pose["knee"]
    ankle = pose["ankle"]
    shoulder = pose["shoulder"]
    head = pose["head"]
    hand = pose["hand"]
    heel = pose["heel"]
    toe = pose["toe"]

    cv2.line(frame, w2p(hip, ox, oy, scale), w2p(knee, ox, oy, scale), color, thickness, cv2.LINE_AA)
    cv2.line(frame, w2p(knee, ox, oy, scale), w2p(ankle, ox, oy, scale), color, thickness, cv2.LINE_AA)
    cv2.line(frame, w2p(heel, ox, oy, scale), w2p(toe, ox, oy, scale), color, thickness, cv2.LINE_AA)
    cv2.line(frame, w2p(hip, ox, oy, scale), w2p(shoulder, ox, oy, scale), color, thickness, cv2.LINE_AA)
    cv2.line(frame, w2p(shoulder, ox, oy, scale), w2p(hand, ox, oy, scale), color, max(1, thickness - 1), cv2.LINE_AA)

    head_radius = int(max(4, 0.085 * scale))
    cv2.circle(frame, w2p(head, ox, oy, scale), head_radius, color, thickness, cv2.LINE_AA)


def draw_left(frame, state, t):
    cv2.rectangle(frame, (0, 0), (LEFT_W, FRAME_H), (28, 30, 38), -1)

    put_text(frame, "PERCEPTION: HAND HEIGHT", 14, 26, (210, 225, 255), 0.60, 1)

    raw = float(state.get("hand_raw", 0.0))
    filt = float(state.get("hand_filtered", 0.0))
    hn = float(state.get("hand_norm", 0.0))

    put_text(frame, f"raw={raw:+.2f}  filtered={filt:+.2f}", 14, 50, (200, 200, 210), 0.50, 1)

    # Simple human side-view representation
    head = (235, 105)
    shoulder = (235, 185)
    hip = (235, 360)

    cv2.circle(frame, head, 26, (190, 195, 205), 2, cv2.LINE_AA)
    cv2.line(frame, (235, 131), hip, (190, 195, 205), 3, cv2.LINE_AA)
    cv2.line(frame, (160, 185), (310, 185), (190, 195, 205), 3, cv2.LINE_AA)
    cv2.line(frame, hip, (195, 480), (190, 195, 205), 3, cv2.LINE_AA)
    cv2.line(frame, hip, (275, 480), (190, 195, 205), 3, cv2.LINE_AA)

    hand_y = int(360 - 250 * np.clip(hn, 0.0, 1.0))
    hand_x = 330

    cv2.line(frame, shoulder, (hand_x, hand_y), (0, 200, 255), 3, cv2.LINE_AA)
    cv2.circle(frame, (hand_x, hand_y), 12, (0, 190, 255), -1, cv2.LINE_AA)

    # Normalized hand-height bar
    bar_x = 58
    top = 120
    bottom = 420

    cv2.rectangle(frame, (bar_x, top), (bar_x + 26, bottom), (90, 95, 105), 2)
    y = int(bottom - (bottom - top) * np.clip(filt, 0.0, 1.0))
    cv2.rectangle(frame, (bar_x, y), (bar_x + 26, bottom), (0, 190, 255), -1)

    put_text(frame, "1.0 overhead", bar_x + 34, top + 10, (180, 185, 195), 0.45, 1)
    put_text(frame, "0.0 hip", bar_x + 34, bottom, (180, 185, 195), 0.45, 1)

    if raw < -0.02:
        put_text(frame, "OVER-DEPTH REQUEST", 14, FRAME_H - 18, (80, 80, 255), 0.55, 1)


def draw_right(frame, state, t, label):
    cv2.rectangle(frame, (LEFT_W, 0), (FRAME_W, FRAME_H), (18, 22, 30), -1)

    pose = state.get("last_pose", None)
    if pose is None:
        pose = emergency_pose()

    ox = 880
    oy = 486
    scale = 250

    put_text(frame, f"{label} | t={t:05.2f}s", LEFT_W + 18, 28, (210, 225, 255), 0.62, 1)
    put_text(
        frame,
        f"Hand={state.get('hand_filtered', 0.0):+.2f}   Pelvis cmd={state.get('command_height', STAND_HEIGHT):.3f} m   actual={state.get('actual_height', STAND_HEIGHT):.3f} m",
        LEFT_W + 18,
        52,
        (200, 205, 215),
        0.48,
        1,
    )

    draw_gauge(
        frame,
        LEFT_W + 18,
        82,
        260,
        18,
        pose.get("torque_pct", 0.0),
        f"Knee torque budget {pose.get('knee_torque', 0.0):.1f} Nm",
    )

    err_mm = float(pose.get("com_error", 0.0)) * 1000.0
    put_text(frame, f"CoM centering error = {err_mm:+.1f} mm", LEFT_W + 18, 126, (0, 230, 255), 0.52, 1)

    if state.get("velocity_limited", False):
        put_text(
            frame,
            "Velocity ramp limit active (0.30 m/s)",
            LEFT_W + 18,
            150,
            (0, 180, 255),
            0.50,
            1,
        )

    msg = pose.get("message", "") or state.get("message", "")
    if msg:
        put_text(frame, msg, LEFT_W + 18, 174, (80, 80, 255), 0.56, 2)

    # Ground
    cv2.line(frame, (LEFT_W, oy), (FRAME_W, oy), (95, 95, 105), 2)

    # Support polygon
    heel_px = w2p(np.array([HEEL_X, 0.0], dtype=float), ox, oy, scale)
    toe_px = w2p(np.array([TOE_X, 0.0], dtype=float), ox, oy, scale)
    cv2.rectangle(frame, (heel_px[0], heel_px[1] - 6), (toe_px[0], heel_px[1]), (70, 140, 70), 2)

    center_px = w2p(np.array([SUPPORT_CENTER, 0.0], dtype=float), ox, oy, scale)
    cv2.drawMarker(frame, center_px, (0, 255, 180), cv2.MARKER_TILTED_CROSS, 20, 2)

    # Fading trail
    history = state.get("history", [])
    n = len(history)

    for i, old in enumerate(history[:-1]):
        f = (i + 1) / max(1, n - 1)
        color = (int(90 * f), int(50 * f), int(150 * f))
        draw_skeleton(frame, old, ox, oy, scale, color, 1)

    draw_skeleton(frame, pose, ox, oy, scale, (220, 200, 90), 2)

    # CoM projection to support polygon
    com = pose.get("com", np.array([SUPPORT_CENTER, 0.8], dtype=float))
    com_px = w2p(com, ox, oy, scale)
    proj_px = w2p(np.array([float(com[0]), 0.0], dtype=float), ox, oy, scale)

    cv2.line(frame, com_px, proj_px, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, com_px, 5, (0, 220, 255), -1, cv2.LINE_AA)


def draw_all(frame, state, t, label):
    frame[:] = (25, 28, 35)
    draw_left(frame, state, t)
    draw_right(frame, state, t, label)


# ----------------------------------------------------------------------------
# Video writing / recording
# ----------------------------------------------------------------------------
def open_video_writer(base_path, fps, size):
    candidates = [
        ("avc1", ".mp4"),
        ("mp4v", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi"),
    ]

    for fourcc, ext in candidates:
        path = base_path if ext == ".mp4" else base_path.replace(".mp4", ext)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path
        writer.release()

    return None, None


def render_video_from_hand(hand_func, duration=DEMO_DURATION, fps=FPS, label="AUTO DEMO"):
    base = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer, path = open_video_writer(base, fps, (FRAME_W, FRAME_H))

    if writer is None:
        return None, "video/mp4"

    state = default_state()
    n = int(duration * fps)

    prog = st.progress(0.0)
    prog_text = f"Rendering {label}..."

    for i in range(n):
        t = i / fps
        hand = hand_func(t)
        step_state(state, hand, 1.0 / fps)

        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        draw_all(frame, state, t, label)
        writer.write(frame)

        if i % (fps * 2) == 0:
            pct = (i + 1) / n
            prog.progress(pct, text=f"{prog_text} {int(pct * 100)}%")

    writer.release()
    prog.empty()

    try:
        data = Path(path).read_bytes()
    except Exception:
        data = None

    for p in {base, path}:
        try:
            os.unlink(p)
        except Exception:
            pass

    mime = "video/mp4" if path.lower().endswith(".mp4") else "video/avi"
    return data, mime


def hand_function_from_log(log, duration=60.0):
    if not log:
        return lambda t: 0.85

    now = time.time()
    start = now - duration

    last_before = float(log[-1][1])
    events = []

    for ts, val in sorted(log):
        if ts < start:
            last_before = float(val)
        else:
            events.append((float(ts - start), float(val)))

    if not events:
        val = float(log[-1][1])
        events = [(0.0, val), (duration, val)]
    else:
        if events[0][0] > 0.0:
            events.insert(0, (0.0, last_before))
        if events[-1][0] < duration:
            events.append((duration, events[-1][1]))

    times = np.array([max(0.0, min(duration, e[0])) for e in events], dtype=float)
    vals = np.array([float(np.clip(e[1], -0.5, 1.2)) for e in events], dtype=float)

    order = np.argsort(times)
    times = times[order]
    vals = vals[order]

    unique = []
    for t, v in zip(times, vals):
        if not unique or t > unique[-1][0]:
            unique.append((float(t), float(v)))
        else:
            unique[-1] = (float(t), float(v))

    ut = np.array([u[0] for u in unique], dtype=float)
    uv = np.array([u[1] for u in unique], dtype=float)

    return lambda t: float(np.interp(t, ut, uv))


def append_manual_log(value):
    now = time.time()
    log = st.session_state.setdefault("manual_log", [])

    value = float(value)

    if not log or abs(log[-1][1] - value) > 1e-4 or now - log[-1][0] > 1.0:
        log.append((now, value))

    cutoff = now - 120.0
    st.session_state.manual_log = [(t, v) for t, v in log if t >= cutoff]


# ----------------------------------------------------------------------------
# Optional live camera / MediaPipe Pose extraction
# ----------------------------------------------------------------------------
def hand_from_pose_landmarks(res):
    if not res or not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark

    lsh, rsh = lm[11], lm[12]
    lhip, rhip = lm[23], lm[24]
    lw, rw = lm[15], lm[16]

    shoulder_y = []
    hip_y = []

    for p in (lsh, rsh):
        if p.visibility > 0.40:
            shoulder_y.append(p.y)

    for p in (lhip, rhip):
        if p.visibility > 0.40:
            hip_y.append(p.y)

    wrist = lw if lw.visibility >= rw.visibility else rw
    if wrist.visibility < 0.35:
        return None

    if not shoulder_y:
        return None

    shoulder = float(np.mean(shoulder_y))
    hip = float(np.mean(hip_y)) if hip_y else shoulder + 0.25

    denom = max(0.05, hip - shoulder)
    hand_raw = (hip - float(wrist.y)) / denom
    return float(hand_raw)


def record_camera_video(duration=60.0):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "video/mp4"

    mp_pose = None
    if mp is not None:
        try:
            mp_pose = mp.solutions.pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            mp_pose = None

    base = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer, path = open_video_writer(base, FPS, (FRAME_W, FRAME_H))

    if writer is None:
        cap.release()
        if mp_pose is not None:
            mp_pose.close()
        return None, "video/mp4"

    state = default_state()
    start = time.time()
    last_frame_time = start

    frame_placeholder = st.empty()
    prog = st.progress(0.0, text="Recording live camera...")

    while True:
        t = time.time() - start
        if t >= duration:
            break

        ret, cam = cap.read()
        if not ret:
            break

        hand_raw = 0.80

        if mp_pose is not None:
            try:
                rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
                res = mp_pose.process(rgb)
                val = hand_from_pose_landmarks(res)
                if val is not None:
                    hand_raw = val
            except Exception:
                pass

        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now
        dt = float(np.clip(dt, 0.02, 0.20))

        step_state(state, hand_raw, dt)

        out = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

        cam_resized = cv2.resize(cam, (LEFT_W, FRAME_H))
        out[0:FRAME_H, 0:LEFT_W] = cam_resized

        draw_right(out, state, t, "LIVE CAMERA")
        put_text(out, f"hand command={hand_raw:+.2f}", 12, 26, (0, 220, 255), 0.6, 1)

        writer.write(out)
        frame_placeholder.image(out, channels="BGR")
        prog.progress(t / duration, text=f"Recording live camera: {t:04.1f}s / {duration:04.1f}s")

    cap.release()
    writer.release()

    if mp_pose is not None:
        mp_pose.close()

    prog.empty()
    frame_placeholder.empty()

    try:
        data = Path(path).read_bytes()
    except Exception:
        data = None

    if data is not None and len(data) == 0:
        data = None

    for p in {base, path}:
        try:
            os.unlink(p)
        except Exception:
            pass

    mime = "video/mp4" if path.lower().endswith(".mp4") else "video/avi"
    return data, mime


# ----------------------------------------------------------------------------
# Streamlit state initialization
# ----------------------------------------------------------------------------
if "manual_state" not in st.session_state:
    st.session_state.manual_state = default_state()

if "last_manual_ts" not in st.session_state:
    st.session_state.last_manual_ts = time.time()

if "manual_log" not in st.session_state:
    st.session_state.manual_log = [(time.time(), 0.85)]

if "demo_video" not in st.session_state:
    st.session_state.demo_video = render_video_from_hand(
        demo_hand_height,
        DEMO_DURATION,
        FPS,
        "AUTO DEMO",
    )

if "live_video" not in st.session_state:
    st.session_state.live_video = None

if "camera_video" not in st.session_state:
    st.session_state.camera_video = None


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("Whole-Body Squat and Crouch Height Controller")
st.markdown(
    "Hand height commands a coordinated hip-knee-ankle crouch while keeping the CoM centered "
    "over the support polygon. The app auto-renders a 60-second demo below and supports live "
    "interactive control."
)

st.sidebar.markdown("### System")
st.sidebar.caption(f"PyBullet available: {pybullet is not None}")
st.sidebar.caption(f"MediaPipe available: {mp is not None}")
st.sidebar.caption("No API key required.")
st.sidebar.caption("Velocity ramp limit: 0.30 m/s")
st.sidebar.caption("CoM target: center of support polygon")

auto_tab, interact_tab, camera_tab = st.tabs(
    [
        "Auto 60-second Demo",
        "Interactive Hand Height",
        "Live Camera / Real Video",
    ]
)

# ----------------------------------------------------------------------------
# Tab 1: automatic 60-second demo
# ----------------------------------------------------------------------------
with auto_tab:
    col_video, col_info = st.columns([2, 1])

    with col_video:
        demo = st.session_state.get("demo_video", (None, "video/mp4"))

        if demo and demo[0] is not None:
            st.video(demo[0], format=demo[1])
            st.download_button(
                "Download 60-second demo video",
                data=demo[0],
                file_name="whole_body_squat_auto_demo.mp4",
                mime=demo[1],
            )
        else:
            st.warning("Video renderer could not open on this system. Showing a static preview instead.")
            state = default_state()
            for i in range(FPS * 3):
                step_state(state, demo_hand_height(i / FPS), 1.0 / FPS)

            frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            draw_all(frame, state, 3.0, "AUTO DEMO PREVIEW")
            st.image(frame, channels="BGR")

        if st.button("Regenerate 60-second demo"):
            st.session_state.pop("demo_video", None)
            st.rerun()

    with col_info:
        st.markdown("### Demo script")
        st.markdown(
            """
            - **0:00 - 0:12**: automatic stand-to-crouch-to-stand cycle  
            - **0:12 - 0:27**: deliberate hand lowering, robot crouches  
            - **0:27 - 0:40**: fast hand raise, velocity ramp limit visible  
            - **0:40 - 0:52**: over-depth command, safety depth clamp active  
            - **0:52 - 1:00**: recovery, CoM centering error remains small  
            """
        )

        st.markdown("### Controller features")
        st.markdown(
            """
            - Coordinated hip-knee-ankle squat IK
            - Support-polygon CoM centering constraint
            - Torque-budget-aware depth limiting
            - 0.30 m/s pelvis-height velocity shaping
            - Torso pitch compensation for natural crouch posture
            - Live CoM-centering error and knee torque gauge
            """
        )

# ----------------------------------------------------------------------------
# Tab 2: interactive control
# ----------------------------------------------------------------------------
with interact_tab:
    st.markdown(
        "Move the slider to command hand height. Check **Push past depth limit** to test the safety clamp. "
        "The preview updates on each Streamlit rerun, and you can render a 60-second video from your recent log."
    )

    col_controls, col_preview = st.columns([1, 2])

    with col_controls:
        hand_slider = st.slider(
            "Hand height command (0 = hip level, 1 = overhead)",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.01,
            key="manual_slider",
        )

        over_depth = st.checkbox("Push past depth limit (-0.35 command)", value=False)
        effective_hand = -0.35 if over_depth else float(hand_slider)

        append_manual_log(effective_hand)

        st.markdown("---")

        if st.button("Render 60-second live-control video from log"):
            with st.spinner("Rendering live-control video..."):
                hand_func = hand_function_from_log(st.session_state.manual_log, 60.0)
                st.session_state.live_video = render_video_from_hand(
                    hand_func,
                    60.0,
                    FPS,
                    "LIVE CONTROL",
                )

        live = st.session_state.get("live_video", None)
        if live and live[0] is not None:
            st.video(live[0], format=live[1])
            st.download_button(
                "Download live-control video",
                data=live[0],
                file_name="whole_body_squat_live_control.mp4",
                mime=live[1],
            )
        elif live is not None:
            st.warning("Live-control video could not be generated on this system.")

    with col_preview:
        now = time.time()
        dt = min(0.25, max(0.03, now - st.session_state.last_manual_ts))
        st.session_state.last_manual_ts = now

        step_state(st.session_state.manual_state, effective_hand, dt)

        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        draw_all(frame, st.session_state.manual_state, now % 60.0, "INTERACTIVE")
        st.image(frame, channels="BGR")

        pose = st.session_state.manual_state["last_pose"]

        m1, m2, m3 = st.columns(3)
        m1.metric("CoM centering error", f"{pose['com_error'] * 1000.0:.1f} mm")
        m2.metric("Knee torque", f"{pose['knee_torque']:.1f} Nm")
        m3.metric("Pelvis height", f"{st.session_state.manual_state['actual_height']:.3f} m")

        if st.session_state.manual_state.get("message", ""):
            st.warning(st.session_state.manual_state["message"])

# ----------------------------------------------------------------------------
# Tab 3: optional live camera recording
# ----------------------------------------------------------------------------
with camera_tab:
    st.markdown(
        """
        Optional real-video mode:

        - Uses webcam index 0 if available.
        - Uses MediaPipe Pose, if installed, to estimate wrist height relative to shoulder/hip.
        - If MediaPipe or body landmarks are not visible, the robot defaults to a safe standing command.
        - Recording duration is 60 seconds.
        """
    )

    if st.button("Record 60-second live camera demo"):
        video = record_camera_video(60.0)
        st.session_state.camera_video = video

    camera_video = st.session_state.get("camera_video", None)

    if camera_video and camera_video[0] is not None:
        st.video(camera_video[0], format=camera_video[1])
        st.download_button(
            "Download live camera video",
            data=camera_video[0],
            file_name="whole_body_squat_camera_demo.mp4",
            mime=camera_video[1],
        )
    elif camera_video is not None:
        st.warning(
            "Camera recording was not available. Check camera permissions or use the Interactive tab instead."
        )
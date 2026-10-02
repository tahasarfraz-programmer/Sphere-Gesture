"""
Galaxy Sphere - hand-gesture controlled 3D particle sphere (+ Saturn ring)
-------------------------------------------------------------------------
Recreated from the laptop video: a cyan/blue dotted 3D sphere with a thin dust ring,
star dust in the background, and a small cyan wireframe of your hand in the corner.

Gestures (one hand, webcam)
  Fist                       -> compact sphere + Saturn ring (slowly spinning)
  Thumb + index "pinch/L"    -> sphere SIZE follows the finger distance (pinch = small, spread = huge)
  Open hand (5 fingers)      -> sphere explodes into drifting blue dust; close the hand to pull it back
  Move hand                  -> sphere follows your hand, left/right motion spins it
  No hand                    -> idle: small sphere in the centre, auto-rotating

Keys:  ESC / Q quit    F fullscreen    H hide / show hand wireframe

Run:
    python sphere_gesture.py            # webcam
    python sphere_gesture.py --demo     # no camera: simulated hand
"""

import math
import os
import sys
import threading
import time

import numpy as np
import pygame

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
WIN_W, WIN_H = 1000, 700
FPS = 60
CAM_INDEX = 0
FLIP_CAMERA = True

BASE_RADIUS = 105.0          # sphere radius in px at scale 1.0
MIN_SCALE, MAX_SCALE = 0.45, 3.4
FOV = 800.0

LAT_RINGS = 40               # latitude rings of the dotted sphere
DOTS_PER_EQUATOR = 84
CORE_POINTS = 700            # solid inner core
RING_POINTS = 2600           # Saturn dust ring
STAR_POINTS = 1300           # background star dust

BG = (6, 9, 16)

MODE_IDLE, MODE_FIST, MODE_SCALE, MODE_OPEN = "idle", "fist", "scale", "open"

HAND_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


# ----------------------------------------------------------------------------
# Hand tracking (runs in a background thread so rendering stays at 60 fps)
# ----------------------------------------------------------------------------
class HandState:
    def __init__(self):
        self.lock = threading.Lock()
        self.present = False
        self.mode = MODE_IDLE
        self.x = 0.5
        self.y = 0.5
        self.pinch = 0.5            # 0..1  (thumb-index distance, normalised)
        self.landmarks = None       # list of (x, y) in 0..1
        self.last_seen = 0.0


class HandTracker(threading.Thread):
    TIPS = [8, 12, 16, 20]
    PIPS = [6, 10, 14, 18]

    def __init__(self, state):
        super().__init__(daemon=True)
        import cv2
        import mediapipe as mp

        self.cv2 = cv2
        self.state = state
        self.running = True
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1, model_complexity=0,
            min_detection_confidence=0.7, min_tracking_confidence=0.6,
        )
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW if os.name == "nt" else 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError("Webcam could not be opened (change CAM_INDEX).")

    @staticmethod
    def _d(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def run(self):
        cv2 = self.cv2
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if FLIP_CAMERA:
                frame = cv2.flip(frame, 1)
            res = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st = self.state
            with st.lock:
                if not res.multi_hand_landmarks:
                    st.present = False
                    continue
                lm = res.multi_hand_landmarks[0].landmark
                size = max(1e-4, self._d(lm[0], lm[9]))
                up = [self._d(lm[t], lm[0]) > self._d(lm[p], lm[0]) * 1.08
                      for t, p in zip(self.TIPS, self.PIPS)]
                thumb_out = self._d(lm[4], lm[17]) > self._d(lm[3], lm[17]) * 1.15
                n_up = sum(up)
                if n_up >= 4 and thumb_out:
                    st.mode = MODE_OPEN
                elif n_up == 0:
                    st.mode = MODE_FIST
                else:
                    st.mode = MODE_SCALE
                pinch = self._d(lm[4], lm[8]) / size
                st.pinch = min(1.0, max(0.0, (pinch - 0.15) / 1.35))
                idx = [0, 5, 9, 13, 17]
                st.x = sum(lm[k].x for k in idx) / 5
                st.y = sum(lm[k].y for k in idx) / 5
                st.landmarks = [(p.x, p.y) for p in lm]
                st.present = True
                st.last_seen = time.time()

    def stop(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


class DemoHand:
    """Simulated hand so you can preview without a camera."""

    def __init__(self, state):
        self.state = state
        self.t0 = time.time()
        self.fan = self._fan()

    @staticmethod
    def _fan():
        pts = [(0.5, 0.9)]
        for f, ang in enumerate([-0.9, -0.35, 0.0, 0.35, 0.7]):
            for k in range(1, 5):
                r = 0.1 + 0.08 * k
                pts.append((0.5 + r * math.sin(ang), 0.9 - r * math.cos(ang)))
        return pts

    def update(self):
        t = time.time() - self.t0
        st = self.state
        with st.lock:
            st.present = True
            st.last_seen = time.time()
            cyc = (t % 16)
            if cyc < 4:
                st.mode = MODE_FIST
            elif cyc < 11:
                st.mode = MODE_SCALE
                st.pinch = 0.5 + 0.5 * math.sin((cyc - 4) * 1.1 - 1.5)
            else:
                st.mode = MODE_OPEN
            st.x = 0.5 + 0.28 * math.sin(t * 0.5)
            st.y = 0.5 + 0.12 * math.cos(t * 0.7)
            st.landmarks = self.fan


# ----------------------------------------------------------------------------
# 3D particle scene
# ----------------------------------------------------------------------------
class SphereScene:
    def __init__(self):
        rng = np.random.default_rng(5)
        self.rng = rng

        # --- dotted latitude/longitude sphere ---
        pts, cols = [], []
        for i in range(1, LAT_RINGS):
            phi = math.pi * i / LAT_RINGS
            r = math.sin(phi)
            y = math.cos(phi)
            n = max(6, int(DOTS_PER_EQUATOR * r))
            off = rng.uniform(0, 2 * math.pi)
            for j in range(n):
                th = 2 * math.pi * j / n + off
                pts.append((r * math.cos(th), y, r * math.sin(th)))
                k = abs(y)
                cols.append((20 + 20 * k, 110 + 115 * k, 255))
        sphere = np.array(pts, dtype=np.float32)
        sphere_col = np.array(cols, dtype=np.float32)

        # --- solid inner core ---
        v = rng.normal(size=(CORE_POINTS, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        core = v * (0.62 * np.cbrt(rng.uniform(0.1, 1, CORE_POINTS)))[:, None]
        core_col = np.tile([25, 95, 225], (CORE_POINTS, 1)).astype(np.float32)

        # --- saturn dust ring (flat in XZ plane) ---
        ang = rng.uniform(0, 2 * math.pi, RING_POINTS)
        rad = 1.5 + 0.42 * np.power(rng.uniform(0, 1, RING_POINTS), 1.15)
        ring = np.stack([rad * np.cos(ang), rng.normal(0, 0.012, RING_POINTS), rad * np.sin(ang)], axis=1)
        ring_col = np.tile([80, 240, 245], (RING_POINTS, 1)).astype(np.float32)
        ring_col *= rng.uniform(0.75, 1.0, (RING_POINTS, 1)).astype(np.float32)

        self.base = np.concatenate([sphere, core, ring]).astype(np.float32)
        self.col = np.concatenate([sphere_col, core_col, ring_col]).astype(np.float32)
        self.kind = np.concatenate([
            np.zeros(len(sphere), np.int8), np.ones(len(core), np.int8), np.full(RING_POINTS, 2, np.int8)
        ])
        self.n = len(self.base)

        # scatter targets (used when the hand is open): px offsets around the sphere centre
        self.scatter = np.stack([
            rng.normal(0, 330, self.n), rng.normal(0, 260, self.n), rng.uniform(-300, 300, self.n)
        ], axis=1).astype(np.float32)
        self.scatter_vel = rng.normal(0, 0.25, (self.n, 3)).astype(np.float32)

        # background stars (screen space, parallax-drift)
        self.stars = np.stack([
            rng.uniform(0, WIN_W, STAR_POINTS), rng.uniform(0, WIN_H, STAR_POINTS)
        ], axis=1).astype(np.float32)
        self.star_v = rng.normal(0, 0.12, (STAR_POINTS, 2)).astype(np.float32)
        self.star_b = rng.uniform(0.25, 1.0, STAR_POINTS).astype(np.float32)

        # state
        self.scale = 1.0
        self.scale_t = 1.0
        self.disp = 0.0
        self.disp_t = 0.0
        self.cx, self.cy = WIN_W / 2, WIN_H / 2
        self.cx_t, self.cy_t = self.cx, self.cy
        self.yaw = 0.0
        self.tilt = 0.35
        self.tilt_t = 0.35
        self.spin = 0.55
        self.t = 0.0

    # ---- controls --------------------------------------------------------------
    def apply_hand(self, st, now):
        with st.lock:
            present = st.present and (now - st.last_seen) < 0.6
            mode, x, y, pinch = st.mode, st.x, st.y, st.pinch
        if not present:
            self.scale_t, self.disp_t = 1.0, 0.0
            self.cx_t, self.cy_t = WIN_W / 2, WIN_H / 2
            self.tilt_t = 0.35
            return MODE_IDLE
        self.cx_t, self.cy_t = x * WIN_W, y * WIN_H
        self.tilt_t = 0.35 + (y - 0.5) * 1.4
        if mode == MODE_FIST:
            self.scale_t, self.disp_t = 1.0, 0.0
        elif mode == MODE_SCALE:
            self.scale_t = MIN_SCALE + (MAX_SCALE - MIN_SCALE) * (pinch ** 1.2)
            self.disp_t = 0.0
        else:  # open hand
            self.disp_t = 1.0
        return mode

    # ---- update ------------------------------------------------------------------
    def update(self, dt):
        self.t += dt
        k = 1 - math.exp(-dt * 7)
        self.scale += (self.scale_t - self.scale) * k
        self.disp += (self.disp_t - self.disp) * (1 - math.exp(-dt * 3.2))
        dx = self.cx_t - self.cx
        self.cx += dx * (1 - math.exp(-dt * 6))
        self.cy += (self.cy_t - self.cy) * (1 - math.exp(-dt * 6))
        self.tilt += (self.tilt_t - self.tilt) * k
        # horizontal hand motion adds spin
        self.yaw += dt * (self.spin + dx * 0.01)

        # drifting dust (only matters when dispersed)
        self.scatter += self.scatter_vel
        for ax, lim in ((0, 650), (1, 520), (2, 320)):
            self.scatter[:, ax] = np.where(np.abs(self.scatter[:, ax]) > lim, -self.scatter[:, ax] * 0.98,
                                           self.scatter[:, ax])
        self.stars += self.star_v
        self.stars[:, 0] %= WIN_W
        self.stars[:, 1] %= WIN_H

    # ---- render ------------------------------------------------------------------
    def render(self, buf):
        buf[:] = BG
        # stars
        sx = self.stars[:, 0].astype(np.int32)
        sy = self.stars[:, 1].astype(np.int32)
        b = (self.star_b * (0.8 + 0.2 * np.sin(self.t * 2 + self.stars[:, 0]))).astype(np.float32)
        for dx_ in (0, 1):
            for dy_ in (0, 1):
                xx = np.minimum(sx + dx_, WIN_W - 1); yy = np.minimum(sy + dy_, WIN_H - 1)
                buf[xx, yy, 0] = (50 * b + BG[0]).astype(np.uint8)
                buf[xx, yy, 1] = (190 * b + BG[1]).astype(np.uint8)
                buf[xx, yy, 2] = (255 * b + BG[2]).astype(np.uint8)

        R = BASE_RADIUS * self.scale
        # rotate: yaw about Y, then tilt about X
        cy_, sy_ = math.cos(self.yaw), math.sin(self.yaw)
        ct, st_ = math.cos(self.tilt), math.sin(self.tilt)
        p = self.base
        x = p[:, 0] * cy_ + p[:, 2] * sy_
        z = -p[:, 0] * sy_ + p[:, 2] * cy_
        y = p[:, 1]
        y2 = y * ct - z * st_
        z2 = y * st_ + z * ct
        x2 = x
        # ring: spins in its own plane, then stands almost edge-on, slanted like the video
        ring = self.kind == 2
        rx, ry, rz = p[ring, 0], p[ring, 1], p[ring, 2]
        a = self.yaw * 0.6
        ca, sa = math.cos(a), math.sin(a)
        rx, rz = rx * ca + rz * sa, -rx * sa + rz * ca            # spin about the ring axis
        ax = 0.34 + 0.10 * math.sin(self.t * 0.45) + (self.tilt - 0.35) * 0.4    # narrow ellipse (near edge-on)
        cax, sax = math.cos(ax), math.sin(ax)
        ry, rz = ry * cax - rz * sax, ry * sax + rz * cax
        az = 1.30 + 0.12 * math.sin(self.t * 0.3)                 # stands up, slanted like the video
        caz, saz = math.cos(az), math.sin(az)
        rx, ry = rx * caz - ry * saz, rx * saz + ry * caz
        x2 = x2.copy(); y2 = y2.copy(); z2 = z2.copy()
        x2[ring], y2[ring], z2[ring] = rx, ry, rz
        x = x2

        world = np.stack([x * R, y2 * R, z2 * R], axis=1)
        d = self.disp
        e = d * d * (3 - 2 * d)
        world = world * (1 - e) + self.scatter * e

        persp = FOV / (FOV + world[:, 2])
        px = (world[:, 0] * persp + self.cx).astype(np.int32)
        py = (world[:, 1] * persp + self.cy).astype(np.int32)
        depth = world[:, 2]

        # brightness: near = bright, far = dim; dispersed dust is dimmer
        bright = np.clip(0.85 - depth / (R * 2.6 + 120), 0.35, 1.2)
        bright *= (1 - 0.2 * e)
        colors = np.clip(self.col * bright[:, None], 0, 255).astype(np.uint8)

        # dot size grows with scale and proximity
        size = np.clip((1.7 + 1.1 * self.scale) * persp, 2.0, 6.0)
        sz = np.where(self.kind == 2, np.maximum(2, size * 0.7), size).astype(np.int32)

        order = np.argsort(-depth)           # far first -> near overwrites
        px, py, colors, sz = px[order], py[order], colors[order], sz[order]
        for s in np.unique(sz):
            m = sz == s
            ox = px[m]; oy = py[m]; c = colors[m]
            for a in range(int(s)):
                for bb in range(int(s)):
                    xx = ox + a
                    yy = oy + bb
                    ok = (xx >= 0) & (xx < WIN_W) & (yy >= 0) & (yy < WIN_H)
                    buf[xx[ok], yy[ok]] = c[ok]


# ----------------------------------------------------------------------------
# Hand wireframe overlay (bottom-right, like the video)
# ----------------------------------------------------------------------------
def draw_hand_overlay(surface, st, now):
    with st.lock:
        lms = st.landmarks
        present = st.present and (now - st.last_seen) < 0.6
    if not present or not lms:
        return
    box = 190
    ox, oy = WIN_W - box - 26, WIN_H - box - 20
    xs = [p[0] for p in lms]
    ys = [p[1] for p in lms]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span = max(maxx - minx, maxy - miny, 1e-3)
    pts = [(ox + (p[0] - minx) / span * box * 0.9 + 8, oy + (p[1] - miny) / span * box * 0.9 + 8) for p in lms]
    col = (60, 205, 255)
    for a, b in HAND_LINES:
        pygame.draw.line(surface, col, pts[a], pts[b], 1)
    for q in pts:
        pygame.draw.circle(surface, (120, 230, 255), (int(q[0]), int(q[1])), 2)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    demo = "--demo" in sys.argv
    pygame.init()
    flags = 0
    screen = pygame.display.set_mode((WIN_W, WIN_H), flags)
    pygame.display.set_caption("Galaxy Sphere - Hand Gesture 3D")
    clock = pygame.time.Clock()

    scene = SphereScene()
    state = HandState()
    buf = np.zeros((WIN_W, WIN_H, 3), dtype=np.uint8)

    tracker = demo_hand = None
    if demo:
        demo_hand = DemoHand(state)
    else:
        tracker = HandTracker(state)
        tracker.start()

    show_hand = True
    running = True
    while running:
        dt = min(0.05, clock.tick(FPS) / 1000.0)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_h:
                    show_hand = not show_hand
                elif e.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()

        if demo_hand:
            demo_hand.update()
        now = time.time()
        scene.apply_hand(state, now)
        scene.update(dt)
        scene.render(buf)
        pygame.surfarray.blit_array(screen, buf)
        if show_hand:
            draw_hand_overlay(screen, state, now)
        pygame.display.flip()

    if tracker:
        tracker.stop()
    pygame.quit()


if __name__ == "__main__":
    main()

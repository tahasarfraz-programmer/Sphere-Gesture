# Nebula Heart 3D

Hand-gesture controlled particle system (OpenCV + MediaPipe + pygame).

## Install (Windows, Python 3.11)
    pip install -r requirements.txt

## Run
    python index.py          # webcam
    python index.py --demo   # no camera, auto-cycles modes

## Gestures
| Gesture | Mode |
|---|---|
| Open hand | BEBAS ANGKASA - blue space dust |
| Fist | BENTUK HATI / LOVE - 3D rotating pink heart |
| Peace (2 fingers) | TEKS: I LOVE YOU - cyan text |
| One finger | SATURNUS 3D - planet with ring |

The shape follows your hand. ESC / Q quits.

---

# Galaxy Sphere (second project: `sphere_gesture.py`)

Cyan/blue 3D dotted sphere + Saturn-style dust ring + star dust, controlled by your hand
(recreated from the laptop video). A small cyan hand wireframe shows in the bottom-right corner.

    python sphere_gesture.py          # webcam
    python sphere_gesture.py --demo   # simulated hand, no camera

| Gesture | Effect |
|---|---|
| Fist | compact sphere + ring, slowly spinning |
| Thumb + index pinch / spread | sphere size follows the finger distance |
| Open hand | sphere explodes into drifting blue dust; close hand to pull it back |
| Move hand | sphere follows it; left/right motion adds spin |
| No hand | idle sphere in the centre |

Keys: ESC/Q quit, F fullscreen, H hide hand wireframe.

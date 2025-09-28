from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class FlowParams:
    t: float  # 0..1 progressive time
    intensity: float  # 0..1 typhoon intensity
    tide_level: float # normalized tide level 0..1
    click_x: float | None = None # 0..1
    click_y: float | None = None # 0..1
    eye_x: float | None = None    # 0..1
    eye_y: float | None = None    # 0..1


def fbm_noise(xy: np.ndarray, seed: int = 0, octaves: int = 4) -> np.ndarray:
    # Simple fractal brownian motion built from sin/cos layers to avoid heavy deps
    x = xy[..., 0]
    y = xy[..., 1]
    val = np.zeros_like(x)
    freq = 1.0
    amp = 0.5
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2*np.pi, size=(octaves, 2))
    for i in range(octaves):
        val += amp * (np.sin(2*np.pi*freq*x + phases[i,0]) * np.cos(2*np.pi*freq*y + phases[i,1]))
        freq *= 2.0
        amp *= 0.5
    # Normalize to 0..1
    val = (val - val.min()) / (val.max() - val.min() + 1e-9)
    return val


def vector_field(grid_x: np.ndarray, grid_y: np.ndarray, p: FlowParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Base curl-like field from fbm noise
    xy = np.stack([grid_x, grid_y], axis=-1)
    base = fbm_noise(xy * (1.5 + 2.0*p.tide_level) + p.t*0.5, seed=int(1000*p.intensity)+7)

    # Compute pseudo-gradient
    gy, gx = np.gradient(base)
    # Curl-ish vector (perpendicular)
    u = -gy
    v = gx

    # Typhoon intensity modulates swirl amplitude
    swirl = 0.8 + 3.2*p.intensity
    u *= swirl
    v *= swirl

    # Inject local vortex on click
    if p.click_x is not None and p.click_y is not None:
        dx = grid_x - p.click_x
        dy = grid_y - p.click_y
        r2 = dx*dx + dy*dy
        influence = np.exp(-r2 / (5 + 5*(1.0-p.tide_level)))
        u += -dy * influence * (0.8 + 0.6*p.intensity)
        v += dx * influence * (0.8 + 0.6*p.intensity)

    # Inject traveling typhoon eye vortex
    if p.eye_x is not None and p.eye_y is not None:
        dx = grid_x - p.eye_x
        dy = grid_y - p.eye_y
        r2 = dx*dx + dy*dy
        # A bit tighter core than click vortex; radius expands slightly at low tide
        influence = np.exp(-r2 / (5 + 10*(1.0-p.tide_level)))
        strength = (2.0 + 3.0*p.intensity)
        u += -dy * influence * strength
        v += dx * influence * strength

    speed = np.sqrt(u*u + v*v)
    speed = speed / (speed.max() + 1e-9)
    return u, v, speed


def advect_particles(pos: np.ndarray, u: np.ndarray, v: np.ndarray, dt: float = 0.01) -> np.ndarray:
    # Bilinear sample u,v from grid
    H, W = u.shape
    x = pos[:, 0] * (W - 1)
    y = pos[:, 1] * (H - 1)

    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, W-1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, H-1)

    sx = x - x0
    sy = y - y0

    def bilinear(field):
        f00 = field[y0, x0]
        f10 = field[y0, x1]
        f01 = field[y1, x0]
        f11 = field[y1, x1]
        return (f00*(1-sx)*(1-sy) + f10*sx*(1-sy) + f01*(1-sx)*sy + f11*sx*sy)

    vel = np.stack([bilinear(u), bilinear(v)], axis=-1)
    pos = pos + dt * vel

    # Wrap-around for a seamless look
    pos[:, 0] = np.mod(pos[:, 0], 1.0)
    pos[:, 1] = np.mod(pos[:, 1], 1.0)
    return pos

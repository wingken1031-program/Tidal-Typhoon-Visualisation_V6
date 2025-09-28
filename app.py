from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from datetime import datetime

from dash import Dash, dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go

from src.data_loader import load_tides, load_typhoons, tide_stats
from src.field import FlowParams, vector_field, advect_particles

ROOT = Path(__file__).resolve().parent

try:
    TIDES = load_tides()  # dict: {'time': [datetime], 'tide': [float]}
    TYPH_BY_NAME = load_typhoons()  # dict: name -> list of rows
except Exception as e:
    TIDES = {'time': [], 'tide': []}
    TYPH_BY_NAME = {}
    print('Data loading error:', e)

unique_typhoons = sorted(TYPH_BY_NAME.keys())

tide_min, tide_max = tide_stats(TIDES)


def tide_norm_at(ts: datetime) -> float:
    if not TIDES['time']:
        return 0.5
    # Nearest tide sample (linear scan; acceptable for interactivity)
    best_i = None
    best_dt = None
    for i, t in enumerate(TIDES['time']):
        dt = abs((t - ts).total_seconds())
        if best_i is None or dt < best_dt:
            best_i = i
            best_dt = dt
    h = float(TIDES['tide'][best_i])
    return (h - tide_min) / (tide_max - tide_min + 1e-9)


# Particle system initial positions
N_PART = 1200
GRID_W, GRID_H = 80, 60
np.random.seed(42)
PARTICLES = np.random.rand(N_PART, 2)

app = Dash(__name__)
app.title = 'Tidal Typhoon Generative Art'


# --- Artistic helpers for eye path ---
def catmull_rom_spline(points, values=None, samples=240):
    # points: list of (x,y) in 0..1; values: optional list same length (e.g., intensity)
    pts = np.asarray(points, dtype=float)
    m = len(pts)
    if m < 2:
        return pts[:,0], pts[:,1], (np.asarray(values) if values is not None else None)
    # Duplicate endpoints to get P0,P1,...,Pm-1 with padding
    P = np.vstack([pts[0], pts, pts[-1]])
    if values is not None:
        V0 = float(values[0])
        Vn = float(values[-1])
        V = np.array([V0, *[float(v) for v in values], Vn], dtype=float)
    else:
        V = None
    segs = m - 1
    t_per_seg = max(1, samples // segs)
    xs, ys, vs = [], [], []
    for i in range(segs):
        P0, P1, P2, P3 = P[i], P[i+1], P[i+2], P[i+3]
        if V is not None:
            v0, v1, v2, v3 = V[i], V[i+1], V[i+2], V[i+3]
        for j in range(t_per_seg + (1 if i == segs-1 else 0)):
            t = j / t_per_seg
            t2, t3 = t*t, t*t*t
            # Catmull–Rom basis matrix with tension 0.5
            a = -0.5*t3 + t2 - 0.5*t
            b =  1.5*t3 - 2.5*t2 + 1.0
            c = -1.5*t3 + 2.0*t2 + 0.5*t
            d =  0.5*t3 - 0.5*t2
            p = a*P0 + b*P1 + c*P2 + d*P3
            xs.append(p[0]); ys.append(p[1])
            if V is not None:
                vs.append(a*v0 + b*v1 + c*v2 + d*v3)
    return np.array(xs), np.array(ys), (np.array(vs) if V is not None else None)


def hsl_color(h, s=80, l=50, a=0.9):
    return f"hsla({int(h)%360}, {int(s)}%, {int(l)}%, {a:.3f})"

# --- Map helpers (centered near Hong Kong) ---
HK_LAT, HK_LON = 22.30, 114.17

def compute_extent(ty_name: str | None):
    # Determine a lat/lon bounding box including HK and the selected storm; enforce minimum span for aesthetics
    min_lat_span = 6.0
    min_lon_span = 8.0
    if ty_name and ty_name in TYPH_BY_NAME:
        sub = TYPH_BY_NAME[ty_name]
        lats = [float(r['lat']) for r in sub if r.get('lat') is not None]
        lons = [float(r['lon']) for r in sub if r.get('lon') is not None]
    else:
        lats, lons = [], []
    if not lats or not lons:
        lat_min, lat_max = HK_LAT - 4, HK_LAT + 4
        lon_min, lon_max = HK_LON - 6, HK_LON + 6
    else:
        lat_min, lat_max = min(min(lats), HK_LAT), max(max(lats), HK_LAT)
        lon_min, lon_max = min(min(lons), HK_LON), max(max(lons), HK_LON)
        # Center and pad to minimum spans
        lat_c = (lat_min + lat_max) / 2.0
        lon_c = (lon_min + lon_max) / 2.0
        lat_span = max(lat_max - lat_min, min_lat_span)
        lon_span = max(lon_max - lon_min, min_lon_span)
        lat_min, lat_max = lat_c - lat_span/2, lat_c + lat_span/2
        lon_min, lon_max = lon_c - lon_span/2, lon_c + lon_span/2
    return float(lat_min), float(lat_max), float(lon_min), float(lon_max)

def norm_from_latlon(lat: float, lon: float, extent):
    lat_min, lat_max, lon_min, lon_max = extent
    lat_rng = (lat_max - lat_min) or 1.0
    lon_rng = (lon_max - lon_min) or 1.0
    x = (lon - lon_min) / lon_rng
    y = 1.0 - (lat - lat_min) / lat_rng
    return float(x), float(y)

def latlon_from_norm(x: float, y: float, extent):
    lat_min, lat_max, lon_min, lon_max = extent
    lat_rng = (lat_max - lat_min) or 1.0
    lon_rng = (lon_max - lon_min) or 1.0
    lon = lon_min + x * lon_rng
    lat = lat_min + (1.0 - y) * lat_rng
    return float(lat), float(lon)

# --- Eye artistry helpers ---
def circle_points(cx: float, cy: float, r: float, n: int = 120):
    th = np.linspace(0, 2*np.pi, n)
    x = cx + r*np.cos(th)
    y = cy + r*np.sin(th)
    return x, y

def spiral_points(cx: float, cy: float, a: float, b: float, th0: float, th1: float, n: int = 140):
    th = np.linspace(th0, th1, n)
    r = a * np.exp(b*th)
    x = cx + r*np.cos(th)
    y = cy + r*np.sin(th)
    return x, y

app.layout = html.Div([
    html.H1('Tidal Typhoon Generative Art'),
    html.Div([
        html.Div([
            html.Label('Typhoon'),
            dcc.Dropdown(id='typhoon', options=[{'label': n, 'value': n} for n in unique_typhoons], value=(unique_typhoons[0] if unique_typhoons else None), clearable=False),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '12px'}),
        html.Div([
            html.Label('Time'),
            dcc.Slider(id='time_idx', min=0, max=100, step=1, value=0, updatemode='drag'),
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.Label('Speed'),
            dcc.Slider(id='speed', min=0.5, max=3.0, step=0.5, value=1.0),
        ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '12px'}),
        html.Div([
            html.Label('Trail'),
            dcc.Slider(id='trail', min=0, max=1, step=0.02, value=0.5, updatemode='drag'),
        ], style={'width': '12%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '12px'}),
        html.Div([
            html.Button('Play', id='play_btn', n_clicks=0, style={'width': '100%', 'height': '38px', 'marginTop': '22px'}),
        ], style={'width': '10%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '12px'}),
    ], id='controls'),

    dcc.Graph(id='graph', style={'height': '80vh'}),

    # Hidden stores for particles and clicks
    dcc.Store(id='particles', data=PARTICLES.tolist()),
    dcc.Store(id='last_click', data=None),
    dcc.Store(id='playing', data=False),
    dcc.Interval(id='tick', interval=250, n_intervals=0, disabled=True),
], style={'padding': '10px 16px'})


@app.callback(
    Output('time_idx', 'max'),
    Output('time_idx', 'value'),
    Input('typhoon', 'value'),
    Input('tick', 'n_intervals'),
    State('time_idx', 'value'),
    State('speed', 'value'),
    State('playing', 'data'),
)
def update_time_range(ty_name: str | None, _n_tick, cur_val, speed, playing):
    if not ty_name or ty_name not in TYPH_BY_NAME:
        return 100, 0
    sub = TYPH_BY_NAME[ty_name]
    max_idx = max(len(sub) - 1, 1)

    # If triggered by typhoon change, reset to 0
    if ctx.triggered_id == 'typhoon':
        return max_idx, 0

    # If ticking and playing, advance
    if ctx.triggered_id == 'tick' and playing:
        step = max(1, int(round(speed or 1.0)))
        next_val = int(((cur_val or 0) + step) % (max_idx + 1))
        return max_idx, next_val

    # Otherwise keep current value
    return max_idx, int(cur_val or 0)


@app.callback(
    Output('playing', 'data'),
    Output('tick', 'disabled'),
    Output('play_btn', 'children'),
    Input('play_btn', 'n_clicks'),
    State('playing', 'data'),
)
def toggle_play(n_clicks, playing):
    if n_clicks is None:
        return False, True, 'Play'
    new_state = not bool(playing)
    return new_state, (not new_state), ('Pause' if new_state else 'Play')


@app.callback(
    Output('graph', 'figure'),
    Output('particles', 'data'),
    Input('time_idx', 'value'),
    Input('typhoon', 'value'),
    Input('trail', 'value'),
    State('particles', 'data'),
    State('last_click', 'data'),
)
def render(idx, ty_name, trail, particles_data, last_click):
    pos = np.array(particles_data) if particles_data is not None else np.random.rand(N_PART, 2)
    click = last_click or {}
    # We'll derive cx,cy (normalized) after computing extent; last_click may store lat/lon
    cx = cy = None

    if ty_name and ty_name in TYPH_BY_NAME:
        sub = TYPH_BY_NAME[ty_name]
        idx = int(np.clip(idx or 0, 0, max(len(sub)-1, 0)))
        row = sub[idx]
        ts = row['time']
        intensity = float(row.get('intensity', 0.0))
        extent = compute_extent(ty_name)
        eye_x = eye_y = None
        if row.get('lat') is not None and row.get('lon') is not None:
            eye_x, eye_y = norm_from_latlon(float(row['lat']), float(row['lon']), extent)
        track_norm = [norm_from_latlon(float(r['lat']), float(r['lon']), extent)
                      for r in sub if r.get('lat') is not None and r.get('lon') is not None]
    else:
        ts = datetime.now()
        intensity = 0.0
        extent = compute_extent(None)
        eye_x = eye_y = None
        track_norm = []

    t_norm = tide_norm_at(ts)

    # Build field
    gx = np.linspace(0, 1, GRID_W)
    gy = np.linspace(0, 1, GRID_H)
    grid_x, grid_y = np.meshgrid(gx, gy)
    # Resolve click location (convert lat/lon to normalized if needed)
    if isinstance(click, dict):
        if 'lat' in click and 'lon' in click:
            cx, cy = norm_from_latlon(float(click['lat']), float(click['lon']), extent)
        elif 'x' in click and 'y' in click:  # backward compatibility
            cx, cy = float(click['x']), float(click['y'])
    p = FlowParams(t=(idx or 0)/max(1,(GRID_W-1)), intensity=intensity, tide_level=t_norm, click_x=cx, click_y=cy, eye_x=eye_x, eye_y=eye_y)
    u, v, speed = vector_field(grid_x, grid_y, p)

    # Advect particles a few sub-steps for smoother trails
    for _ in range(3):
        pos = advect_particles(pos, u, v, dt=0.02 + 0.06*intensity)

    # Determine subset to draw; compute colors for that subset to keep lengths consistent
    keep = int(N_PART * (0.4 + 0.55*trail))
    keep = int(np.clip(keep, 200, N_PART))
    pos_draw = pos[:keep]
    hue_draw = speed[(pos_draw[:,1]*(GRID_H-1)).astype(int), (pos_draw[:,0]*(GRID_W-1)).astype(int)]
    colors_draw = [f'hsl({int(220 + 100*c)%360}, 80%, {int(40 + 35*c)}%)' for c in hue_draw]

    lat_draw = []
    lon_draw = []
    for xy in pos_draw:
        lat_i, lon_i = latlon_from_norm(float(xy[0]), float(xy[1]), extent)
        lat_draw.append(lat_i)
        lon_draw.append(lon_i)

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=lon_draw, lat=lat_draw, mode='markers',
        marker=dict(size=2, color=colors_draw, opacity=0.85),
        hoverinfo='none', showlegend=False
    ))

    # Draw artistic spline path and eye with comet glow
    if track_norm:
        # Get intensity per track point (fallback 0.5)
        sub = TYPH_BY_NAME[ty_name]
        vals = [float(r.get('intensity', 0.5)) for r in sub if r.get('lat') is not None and r.get('lon') is not None]
        tx, ty, ti = catmull_rom_spline(track_norm, values=vals, samples=300)
        # Layered glow: widest faint, then narrower brighter
        for k, width, alpha in (
            (0, 10, 0.10),
            (1, 6,  0.18),
            (2, 3,  0.28),
        ):
            hue = 200 + 120*(ti if ti is not None else t_norm)  # blue->cyan with intensity
            col = hsl_color(hue.mean() if hasattr(hue, 'mean') else hue, 80, 55 - 10*k, alpha)
            # Convert spline points back to lat/lon for drawing on geo
            lat_line = []
            lon_line = []
            for x_i, y_i in zip(tx, ty):
                la, lo = latlon_from_norm(float(x_i), float(y_i), extent)
                lat_line.append(la)
                lon_line.append(lo)
            fig.add_trace(go.Scattergeo(
                lon=lon_line, lat=lat_line, mode='lines',
                line=dict(color=col, width=width),
                hoverinfo='skip', showlegend=False
            ))
    if eye_x is not None and eye_y is not None:
        # Eye marker with small comet tail (short segment behind current index)
        eye_lat, eye_lon = latlon_from_norm(eye_x, eye_y, extent)
        # Pulsing eyewall: concentric rings with tide/intensity modulation
        base_r = 0.018 * (1.0 + 0.6*(1.0 - t_norm))
        ring_scales = [0.8, 1.0, 1.25]
        ring_alpha = [0.10, 0.18, 0.25]
        for scale, alpha in zip(ring_scales, ring_alpha):
            rx, ry = circle_points(eye_x, eye_y, base_r*scale, n=120)
            rlats, rlons = [], []
            for x_i, y_i in zip(rx, ry):
                la, lo = latlon_from_norm(float(x_i), float(y_i), extent)
                rlats.append(la); rlons.append(lo)
            fig.add_trace(go.Scattergeo(
                lon=rlons, lat=rlats, mode='lines',
                line=dict(color='rgba(255,230,160,%.3f)'%alpha, width=2),
                hoverinfo='skip', showlegend=False
            ))
        # Spiral rainbands (two arms), length/intensity scaled
        arms = 2
        for arm in range(arms):
            th0 = arm*np.pi
            th1 = th0 + (2.5 + 2.0*intensity)
            spx, spy = spiral_points(eye_x, eye_y, a=0.002, b=0.22 + 0.2*intensity, th0=th0, th1=th1, n=140)
            slats, slons = [], []
            for x_i, y_i in zip(spx, spy):
                la, lo = latlon_from_norm(float(x_i), float(y_i), extent)
                slats.append(la); slons.append(lo)
            fig.add_trace(go.Scattergeo(
                lon=slons, lat=slats, mode='lines',
                line=dict(color='rgba(180,210,255,0.25)', width=2),
                hoverinfo='skip', showlegend=False
            ))
        # Eye core marker
        fig.add_trace(go.Scattergeo(
            lon=[eye_lon], lat=[eye_lat], mode='markers',
            marker=dict(size=10, color='rgba(255,220,120,0.95)', line=dict(color='rgba(255,255,255,0.9)', width=1)),
            hoverinfo='skip', showlegend=False
        ))
        if track_norm:
            # Tail from recent segment of spline
            # Approximate current param along spline from idx
            sub = TYPH_BY_NAME[ty_name]
            vals = [float(r.get('intensity', 0.5)) for r in sub if r.get('lat') is not None and r.get('lon') is not None]
            tx, ty, ti = catmull_rom_spline(track_norm, values=vals, samples=300)
            # Map discrete idx to spline index proportionally
            s_idx = int((idx / max(1, len(sub)-1)) * (len(tx)-1))
            tail_len = max(6, int(24 * (0.4 + 0.6*intensity)))
            s0 = max(0, s_idx - tail_len)
            hue_tail = 220 + 100*(ti[s0:s_idx+1].mean() if ti is not None else intensity)
            for k, width, alpha in ((0, 12, 0.10), (1, 7, 0.18), (2, 4, 0.30)):
                col = hsl_color(hue_tail, 80, 60 - 10*k, alpha)
                lat_tail = []
                lon_tail = []
                for x_i, y_i in zip(tx[s0:s_idx+1], ty[s0:s_idx+1]):
                    la, lo = latlon_from_norm(float(x_i), float(y_i), extent)
                    lat_tail.append(la)
                    lon_tail.append(lo)
                fig.add_trace(go.Scattergeo(
                    lon=lon_tail, lat=lat_tail, mode='lines',
                    line=dict(color=col, width=width),
                    hoverinfo='skip', showlegend=False
                ))
            # Heading arrow from last to current eye
            if s_idx > 2:
                la0, lo0 = latlon_from_norm(float(tx[s_idx-2]), float(ty[s_idx-2]), extent)
                la1, lo1 = eye_lat, eye_lon
                fig.add_trace(go.Scattergeo(
                    lon=[lo0, lo1], lat=[la0, la1], mode='lines',
                    line=dict(color='rgba(255,255,255,0.5)', width=2),
                    hoverinfo='skip', showlegend=False
                ))

    # Configure world atlas (geo) extents and style
    lat_min, lat_max, lon_min, lon_max = extent
    fig.update_geos(
        projection_type='natural earth',
        showcountries=True,
        countrycolor='rgba(170,190,240,0.6)',
        showcoastlines=True, coastlinecolor='rgba(190,210,255,0.55)',
        showland=True, landcolor='rgba(18,26,50,1.0)',
        showocean=True, oceancolor='rgba(5,10,25,1.0)',
        lakecolor='rgba(5,10,25,1.0)',
        lataxis_range=[lat_min, lat_max],
        lonaxis_range=[lon_min, lon_max],
        bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(
        showlegend=False,
        paper_bgcolor='#0b1020',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=20,b=0),
        title=f"{ty_name or 'No Typhoon'} — {ts.strftime('%Y-%m-%d %H:%M')} — Tide {t_norm:.2f} — Intensity {intensity:.2f}"
    )
    # Labels as text traces
    fig.add_trace(go.Scattergeo(lon=[HK_LON], lat=[HK_LAT], mode='text',
                                text=['Hong Kong'], textfont=dict(color='rgba(220,235,255,0.9)', size=12),
                                hoverinfo='skip', showlegend=False))
    scs_lat, scs_lon = 15.0, 116.0
    fig.add_trace(go.Scattergeo(lon=[scs_lon], lat=[scs_lat], mode='text',
                                text=['South China Sea'], textfont=dict(color='rgba(180,200,255,0.7)', size=12),
                                hoverinfo='skip', showlegend=False))

    return fig, pos.tolist()


@app.callback(
    Output('last_click', 'data'),
    Input('graph', 'clickData')
)
def on_click(clickData):
    if not clickData:
        return None
    # For geo plots, points provide 'lat' and 'lon'
    pt = clickData['points'][0]
    if 'lat' in pt and 'lon' in pt:
        return {'lat': float(pt['lat']), 'lon': float(pt['lon'])}
    # Fallback to cartesian
    if 'x' in pt and 'y' in pt:
        return {'x': float(pt['x']), 'y': float(pt['y'])}
    return None


if __name__ == '__main__':
    app.run_server(debug=True)

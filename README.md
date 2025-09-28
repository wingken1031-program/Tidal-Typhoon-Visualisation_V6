# Tidal Typhoon Generative Art

An interactive Dash app that visualizes and sonifies the dynamic relationship between Hong Kong tides and typhoon passages. Select a typhoon to morph a tide field and drive particle flows.

## Features
- Load tides (2022–2024) and typhoon tracks from CSVs in this folder.
- Dropdown to select typhoon.
- Time slider to scrub through a storm.
- Generative particle field whose flow warps with wind/rain/pressure proxies.
- Interactive: click to inject a local vortex, hover to modulate noise.

## Quickstart (Windows PowerShell)

1. Create a virtual environment (optional but recommended):
```
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the app:
```
python app.py
```

Then open the URL shown in the terminal (usually http://127.0.0.1:8050/).

## CSVs
- `2022-2024_Hong_Kong_Tidal_Report.csv`
- `Typhoon_data.csv`

These are expected to include timestamps and typhoon names/metrics. The app tries to infer common column names. If your columns differ, tweak `src/data_loader.py` mappings.

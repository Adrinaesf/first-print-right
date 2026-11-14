# Slicer training data

The bundled `data/slicer-training.json` contains **192 real PrusaSlicer estimates**, from **24 procedural solids × 8 orientations**: boxes, L brackets, T profiles, and tapered cylinders, each at six sizes. The generator makes closed solids using Three.js, rotates them before export, and slices each pose separately. Six axial poses plus two oblique poses include both flat contact and point/edge contact cases.

These are **slicer-labelled demonstration data**, not measured physical prints. The small collection cannot establish real-world accuracy, print success probability, or generalization to arbitrary uploaded parts. Related size variants remain correlated even though train/holdout model IDs are separated. Failure risk remains the existing geometric heuristic; no failure classifier has been trained.

## Reproduce

Download **PrusaSlicer 2.9.7** from the [official release](https://github.com/prusa3d/PrusaSlicer/releases/tag/version_2.9.7). The generator intentionally rejects other versions so a changed slicer cannot silently change the dataset.

On macOS, point `PRUSA_SLICER` to the executable inside the installed app (or its mounted official DMG), then run from the project directory:

```sh
PRUSA_SLICER='/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer' node scripts/generate-slicer-data.mjs
```

On Linux, use the path to the PrusaSlicer 2.9.7 executable. This repository does not include the slicer binary. No printer is contacted and no G-code is sent to a printer. Temporary STL and G-code files are removed after generation. Set `KEEP_SLICER_OUTPUT=1` to retain them for inspection; the printed temporary path identifies the files.

The generator fails if slicing fails, print-time metadata is missing, or the slicer version differs. It never substitutes heuristic labels. The committed expanded `scripts/slicer-profile.ini` fixes the slicing configuration; its SHA-256 is included in dataset provenance.

## Fixed profile

- PLA density **1.24 g/cm³**, **1.75 mm** filament, **0.4 mm** nozzle/extrusion width.
- **0.2 mm** first and subsequent layers, **3 perimeters**, **6 top and bottom solid layers**, **15% rectilinear infill**.
- Automatic **grid supports**, **45° threshold**, **2.27 mm clear support spacing** (~15% nominal density at 0.4 mm extrusion width), **0 interface layers**, **0.2 mm Z gap**.
- **No brim, skirt, or raft**. This is not tree support.
- Fixed feature speeds: **40 mm/s** perimeter/support, **30 mm/s** external perimeter/top, **60 mm/s** infill, **150 mm/s** travel, **20 mm/s** first layer; cooling slowdown disabled.
- Build volume **220 × 220 × 250 mm**. Electricity is derived using the app's **120 W** assumption; it is not a learned or measured target.

Additional machine and slicing defaults are frozen in the expanded profile. The profile is a generic calibration configuration, not a validated physical printer preset.

## Targets

`hours` is PrusaSlicer's **normal-mode estimated printing time**, converted from the G-code footer to hours. It includes slicer-estimated feature and travel behavior, not measured elapsed print time.

`supportG` sums positive deposited filament extrusion on XY moves tagged `;TYPE:Support material` or `;TYPE:Support material interface`. Relative and absolute extrusion commands are interpreted; stationary retract/unretract moves are excluded. Filament length converts to grams using the fixed 1.75 mm diameter and PLA density. Support interface is disabled in the bundled profile. The parser accepts normal G-code decimal forms including `E.078` and fails when duration metadata is absent.

## Dataset contract

```json
{
  "schemaVersion": 1,
  "provenance": {"kind": "slicer", "description": "Dataset source and limitations"},
  "profile": {"material": "PLA", "layerMm": 0.2, "infill": 0.15, "supportDensity": 0.15, "wallMm": 1.2, "printerWatts": 120},
  "rows": [{
    "modelId": "bracket-2",
    "features": {
      "volumeMm3": 1000, "surfaceMm2": 600,
      "overhangMm2": 100, "contactMm2": 50, "heightMm": 20,
      "supportProxyMm3": 1500, "footprintMm2": 200
    },
    "targets": {"supportG": 0.3, "hours": 0.2},
    "baseline": {"supportG": 0.28, "hours": 0.15}
  }]
}
```

The numeric row above illustrates the schema and is not a training observation. Compute features using the shared `public/analysis.js` helpers before slicing the identical pose. `baseline` stores the original heuristic predictions for honest holdout comparison. All orientations of a source model share `modelId` so the trainer splits by model, never by individual pose.

For a stronger model, collect a larger, diverse, licensed STL corpus under this same profile and validate on entirely unseen shape families. To predict actual failure probability, separately collect physical print outcomes, printer/material settings, and repeated trials; slicer durations and support estimates cannot supply those labels.

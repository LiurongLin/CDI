# Script Layout

Standalone utility scripts are grouped by function:

- `scripts/fits/`: FITS cube assembly, deduplication, and frame-axis reduction.
- `scripts/frequency_maps/`: FFT/windowed map builders and band-limited reconstructions.
- `scripts/frequency_maps/ring_planet_data_processing_pipeline.py`: end-to-end pipeline for combining ring-planet cubes, extracting aperture time series, plotting FFTs, and building coherence/incoherence maps.
- `scripts/regions/`: region definitions, aperture-sum extraction, and annulus-region analysis.
- `scripts/visualization/`: plotting, contact-sheet rendering, GIF export, and FFT visualization helpers.
- `scripts/coronagraph/`: direct coronagraph entrypoints and thin app-level launchers.

There are no root-level standalone `*.py` scripts anymore; use the grouped paths under `scripts/`.

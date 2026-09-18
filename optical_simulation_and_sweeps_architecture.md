# Optical Simulation And Sweep Execution Architecture

This view focuses on two connected layers:

1. the optical propagation core in `coronagraph/simulator.py`
2. the parameter-sweep orchestration in `coronagraph/roddier_sweeps.py` and `coronagraph/cli.py`

```mermaid
flowchart TD
    subgraph SweepExecution["Sweep execution"]
        CLI["coronagraph/cli.py"] --> SWEEPS["coronagraph/roddier_sweeps.py"]
        CLI --> FEATURE["coronagraph/cdi_feature.py"]
        FEATURE --> WORKFLOWS["coronagraph/cdi_workflows.py"]

        SWEEPS --> MASKS["coronagraph/masks.py"]
        SWEEPS --> REGIONS["coronagraph/region_shapes.py"]
        SWEEPS --> SIM["coronagraph/simulator.py"]

        WORKFLOWS --> ANALYSIS["coronagraph/cdi_analysis.py"]
        WORKFLOWS --> REPORTS["coronagraph/cdi_reports.py"]
        WORKFLOWS --> REGIONS
        WORKFLOWS --> SIM
        FEATURE --> ANALYSIS
        FEATURE --> REPORTS
        FEATURE --> REGIONS
        FEATURE --> SIM
    end

    subgraph OpticalCore["Optical simulation core"]
        SIM --> PUPIL["Entrance pupil build<br/>obscuration + spiders + supersampling"]
        SIM --> SCREEN["Optional phase screen<br/>load + resample + cache"]
        SIM --> SHIFT["Optional focal shift<br/>pupil phase ramp"]

        PUPIL --> FFT1["FFT to first focal plane"]
        SCREEN --> FFT1
        SHIFT --> FFT1

        FFT1 --> PHASEMASK["Sampled focal-plane mask<br/>from masks.py"]
        FFT1 --> LOCALPHASE["Global/local focal phase map"]

        PHASEMASK --> FOCALAFTER["Masked focal field"]
        LOCALPHASE --> FOCALAFTER

        FOCALAFTER --> IFFT1["Inverse FFT to Lyot plane"]
        IFFT1 --> LYOT["Lyot stop"]
        LYOT --> FFT2["FFT to final focal plane"]
        FFT2 --> CORON["Normalized coronagraphic PSF"]

        FFT1 --> DIRECT["Normalized direct PSF"]

        DIRECT --> GHOST["Ghost branch<br/>direct/coronagraphic/phase-mask-refraction seed"]
        CORON --> GHOST
        LYOT --> GHOST
        GHOST --> COMBINE["Combine coronagraphic + ghost + interference"]

        CORON --> COMBINE
        COMBINE --> COMPANION{"Companion enabled?"}
        COMPANION -->|no| RESULT["Result dict"]
        COMPANION -->|yes| CLONE["Clone single-source simulator run<br/>for off-axis companion"]
        CLONE --> RESULT
    end

    SWEEPS --> OpticalCore
    ANALYSIS --> OpticalCore
```

## What Runs Where

- `cli.py` is the outer entrypoint.
- Simple sweeps such as:
  - `sweep_roddier_radius_for_peak_match`
  - `sweep_roddier_phase_for_peak_match`
  - `sweep_local_region_phase_peaks`
  live in `roddier_sweeps.py`.
- CDI-oriented multi-step sweeps live one layer higher in:
  - `cdi_feature.py`
  - `cdi_workflows.py`
  - `cdi_analysis.py`
  - `cdi_reports.py`

## Optical Simulation Pipeline

`CoronagraphSimulator.run()` is the central propagation pipeline:

- build entrance pupil
- apply optional pupil phase screen
- apply optional focal shift as a pupil phase ramp
- FFT into first focal plane
- apply focal-plane phase mask and optional local/global phase offsets
- inverse FFT into Lyot plane
- apply Lyot stop
- FFT into final focal plane
- normalize direct and coronagraphic PSFs
- build ghost contribution
- optionally add coherent interference
- optionally recurse once for the off-axis companion branch
- return a result dictionary with PSFs, fields, masks, and metadata

## Sweep Pattern

Most sweep functions follow the same pattern:

1. build a local `sim_kwargs` variant
2. vary one control parameter or geometry
3. run `CoronagraphSimulator(**local_kwargs).run()`
4. extract a metric or map
5. collect arrays over the sweep
6. optionally save plots or summary artifacts

Concrete examples:

- `sweep_roddier_radius_for_peak_match`:
  varies `RoddierPhaseMask(radius_lamD=...)`
- `sweep_roddier_phase_for_peak_match`:
  varies `RoddierPhaseMask(phase_rad=...)`
- `sweep_local_region_phase_peaks`:
  varies local or global phase injection and measures region peaks
- CDI workflow sweeps:
  vary ROI size, ring geometry, planet position, or rotation and then pass the resulting stacks through CDI analysis/report generation

## Practical Dependency Read

- If you are changing physics or propagation, start in `simulator.py`.
- If you are changing mask behavior, start in `masks.py`.
- If you are changing how a one-parameter sweep is executed, start in `roddier_sweeps.py`.
- If you are changing CDI batch experiments and generated outputs, start in `cdi_feature.py` and `cdi_workflows.py`.

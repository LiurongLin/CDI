# CDI Module Dependency Tree

This is the smaller module dependency view for the CDI path only, derived from the current `coronagraph/` imports.

```mermaid
flowchart TD
    CLI["coronagraph/cli.py<br/>CDI entrypoint"] --> FEATURE["coronagraph/cdi_feature.py<br/>orchestration"]

    FEATURE --> WORKFLOWS["coronagraph/cdi_workflows.py<br/>batch sweeps"]
    FEATURE --> ANALYSIS["coronagraph/cdi_analysis.py<br/>metrics + map construction"]
    FEATURE --> REPORTS["coronagraph/cdi_reports.py<br/>PDF/GIF export"]
    FEATURE --> PLOTTING["coronagraph/plotting.py<br/>figures + CDI map helpers"]
    FEATURE --> REGIONS["coronagraph/region_shapes.py<br/>ROI geometry"]
    FEATURE --> SIM["coronagraph/simulator.py<br/>optical propagation"]

    WORKFLOWS --> ANALYSIS
    WORKFLOWS --> REPORTS
    WORKFLOWS --> PLOTTING
    WORKFLOWS --> REGIONS
    WORKFLOWS --> SIM

    ANALYSIS --> PLOTTING
    ANALYSIS --> REGIONS
    ANALYSIS --> SIM

    REPORTS --> REGIONS
    REPORTS --> SIM

    CDIUTIL["coronagraph/cdi.py<br/>standalone CDI utilities"] --> REGIONS
    CDIUTIL --> SIM
```

## Read It Top-Down

- `cli.py` enters the CDI flow through `cdi_feature.py`.
- `cdi_feature.py` is the main coordinator. It pulls together:
  - sweep execution from `cdi_workflows.py`
  - metric and incoherence-map logic from `cdi_analysis.py`
  - output generation from `cdi_reports.py`
  - plotting helpers from `plotting.py`
  - shared geometry from `region_shapes.py`
  - optical simulation from `simulator.py`
- `cdi_workflows.py` is a narrower orchestration layer for repeated sweep/report jobs.
- `cdi_analysis.py` is the analytical core for CDI scoring and map generation.
- `cdi_reports.py` stays on the export side and depends only on shared geometry/simulation primitives.
- `cdi.py` is adjacent CDI support code, but it is not part of the main `cli.py -> cdi_feature.py` execution path.

## Scope

Excluded on purpose:

- `coc_*` wrapper modules
- GUI code
- scripts under `scripts/`
- tests
- poster asset generation under `spie_poster/`

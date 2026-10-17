Auxiliary absolute-residual-margin comparator for section S4

The two CSVs report 432 spatial configurations on the fixed auxiliary 208-position component grid at daylight backgrounds of 200 and 500 lx: 416 values per configuration. They use the fitting-residual daylight margin from spatial_calibration. They are separate from the 69-point task-metric protection and the independent dense final-validation analysis in analysis_results.

Recalculate and check against these stored values with: python Study/check_daylight_margin.py, run from the Repository root. Optionally supply --output NEW_DIRECTORY to export the recalculated tables without changing this archive. Input files and numerical precision are unchanged.

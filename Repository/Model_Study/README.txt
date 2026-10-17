Numerical and replay supplement
This subproject supplies the supplementary optical and replay analyses, not the primary observed endpoints. Run its commands from this Model_Study directory. See ../README.txt for the observation-led entry point.

Sparse commissioning of four-circuit classroom lighting
Code and data release BSERT-CODE-20261017-C22-MODEL

This archive accompanies Effects of commissioning with sparse measurement points on the performance of four-circuit lighting control in existing classrooms, manuscript release BSERT-SUBMISSION-20261017-S19. It provides fixed analysis inputs, reproducible calculations and a single supported command line.

Quick start
Use Python 3.12. From this Model_Study directory:
python -m venv .venv
Windows PowerShell: .\.venv\Scripts\python.exe -m pip install -r requirements_all.txt
macOS/Linux: .venv/bin/python -m pip install -r requirements_all.txt
Run the following commands using the Python executable of that environment.
python run_study.py info
python run_study.py check-files
python run_study.py verify
python run_study.py reproduce --workers 4 --destination ../../model_analysis_output
python run_study.py draw --destination ../../model_figure_output
python documentation/verify_data_catalog.py

Commands
info reports versions and release metadata. check-files needs only the Python standard library and checks every file in file_manifest.csv. verify performs the 68 numerical invariant checks in an isolated copy. reproduce regenerates all 602 current result files, records SHA256 agreement and separately compares scientific content. Add --strict-bytes only when exact SHA256 agreement is required. The default checks scientific equality with the tolerance below; floating-point differences at machine precision can arise from array storage layout even on the same host. Without that flag, floating values use absolute and relative tolerances of 1e-12; integers, flags, categorical labels, field names, array shapes and dtypes remain exact. The archived files are never rounded or rewritten. draw reads the sixteen existing plotting CSVs and generates all eleven figures without rerunning the numerical study; --figure 1–7 selects numerical-supplement Figures M1–M7, and --figure 8–11 selects S1–S4.
To check an existing generated directory without recalculating: python run_study.py compare --generated ../../model_analysis_output. Add --strict-bytes only when byte identity is required in addition to scientific equality.
Output directories must be new. Default outputs go under runs. Existing files and the supplied archive are not overwritten by the supported entry point. Use run_study.py to run the analysis modules with isolated output directories.

Environment
Numerical requirements: NumPy 2.3.5, Numba 0.68.0 and llvmlite 0.50.0. Figure requirements: NumPy 2.3.5 and Matplotlib 3.11.2. Times New Roman must be installed; on another system provide a licensed font folder with --font-dir. Font files are not distributed. Validation was run on Windows with Python 3.12.14. Other platforms are not asserted to produce identical numerical hashes until tested.
See documentation/ENVIRONMENT_AND_RUNTIME.txt for execution evidence and resource guidance. The relative example output paths above are siblings of Repository when run from Model_Study; they must not already exist. documentation/verify_data_catalog.py checks all indexed data-file hashes, sizes, field coverage, publication paths and catalog summary counts without modifying data.

Archive navigation
Study/input_storage_map.json locates the fixed verification and protocol inputs in their canonical storage. Study/analysis_results and Study/camera_reference hold the matching results and camera reference. Study/spatial_calibration contains the fitting pool; Study/light_field_inputs contains the light-field and validation inputs; Study/metering_data contains the metering example. Study/paper_table_index_en.csv preserves the complete 49-entry figure/table index. documentation/paper_output_map.csv adds repository paths, roles and computing stages. documentation/data_dictionary.csv describes every CSV column, NPZ array and JSON field family; data_files.csv records the individual data-file paths and sizes. documentation/code_inventory.csv and dependency_graph.json identify the active imports.

Shared storage
Keep the complete Repository folder together. Study/input_storage_map.json maps logical input names to their single stored copies and records sizes and SHA256 hashes. Fixed input files identical to archived analysis files use those copies; shared metering files use ../Observed_Study/data/metering. The supported runner verifies these mappings and stages complete inputs into a fresh working directory before calculations. It does not modify the archived inputs or results. Input roles are unchanged by their storage locations.
PNG figure previews are retained once. Main and model Figure 1 share ../Observed_Study/data/Figure_01.png; the original camera photograph used by the model drawing program is in Study/figure_redraw. Vector exports are generated by draw rather than stored repeatedly. Journal-ready independent TIFF/EPS files are supplied in the separate submission package.

Analysis inputs and evidence roles
The analysis reads fixed commissioning, protection, terminal, sequential-test and energy records. Study/input_storage_map.json and the remaining Study/fixed_inputs identify the terminal, protection, sequential and precision-test inputs and twelve prescribed controlled-trial profiles. Study/spatial_calibration holds fitting observations. Shared cumulative-energy and execution records are stored in ../Observed_Study/data/metering; Study/metering_data retains numerical inputs and derived metering results. documentation/analysis_input_roles.csv identifies canonical storage paths and input roles. No instrument driver is required for reanalysis of these tabulated records.
The trial profiles are prescribed test inputs, not a sample of naturally occurring building use. Controller outputs for alternative policies are paired replays. true_H and true_day arrays are numerical benchmark components, and prior and direct_day arrays are optical basis terms. All-day quality against those components is a calculated endpoint. Reconstruction from terminal observations is a separate sensitivity path. Numerical noise, optical-mismatch, camera-response, temporal/shape and precision scenarios retain their stated models and assumptions. These quantities are not additional measured field sessions.

Publication and citation
The authors are Zhongshuai Li, Yue Song, Yuxin Yan, Zhenhua Fan, Nanjie Zhou, Longhang Zhou, Zhejie Meng. All authors are affiliated with School of Energy and Power Engineering, Changchun Institute of Technology, Changchun 130012, China. The corresponding author is Yue Song (yue_song@live.cn; ORCID https://orcid.org/0009-0008-7104-5368). The repository is https://github.com/thats0n9/four-circuit-lighting-control. Software is licensed under MIT and data under CC BY 4.0; rights holders are Zhongshuai Li and Yue Song. See ../LICENSE, ../LICENSE_DATA.txt and LICENSING_STATUS.txt. Repository and article DOIs have not been assigned. CITATION_TEMPLATE.txt and LICENSING_STATUS.txt identify these citation and licensing fields. Full author email addresses and funding information appear in ../README.txt and release.json.

Auxiliary comparator
Study/auxiliary_diagnostics supplies the absolute-residual daylight comparator reported in section S4. Check it with python Study/check_daylight_margin.py. These results use their own 208-position component grid and do not replace the independent research-acceptance results.

Publication label mapping
documentation/publication_code_aliases.csv maps the stored DU code in the numerical analysis and model-supplement Figure M6 to DZU with global repair, and lists the spatial-layout and model labels. Stored numerical and categorical values are retained.

Stored comparison labels
In the common-reference binning comparison, the stored family codes legacy and current denote uniform-margin and task-metric protection, respectively. They are algorithm-family labels, not manuscript versions. Their mappings are recorded in documentation/publication_code_aliases.csv. The stored CSV values and NPZ keys are retained for exact reproduction.

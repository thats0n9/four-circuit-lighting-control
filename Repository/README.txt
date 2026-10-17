Classroom lighting commissioning data and code
Release BSERT-CODE-20261017-C22
Related manuscript BSERT-SUBMISSION-20261017-S19

Primary analysis
Observed_Study/data holds the fixed illuminance readings, register endpoints, applied-output logs and instrument settings for classroom R16. The primary endpoints are independent snapshot errors and task classifications, and end-minus-start cumulative electricity on 48 allocated days. State-duration integration is a separate accounting diagnostic. Continuous task-plane lighting quality was not recorded on those days.

Quick start from Repository with Python 3.12
python -m pip install -r requirements.txt
python run_study.py info
python run_study.py check-files
python run_study.py verify
python run_study.py analyse --destination ../observed_analysis_output
python run_study.py draw --destination ../main_figure_output
Analysis and drawing require new destinations outside Repository. NumPy 2.3.5 supports analysis; drawing additionally requires Matplotlib 3.11.2 and installed Times New Roman. Font files are not distributed.

Analysis interfaces
observed_analysis.py fits responses and analyses meter records. mechanism_analysis.py derives snapshot decision matrices, 45 request/background combinations and recorded-output accounting. execution_eligible identifies records suitable for output diagnostics; actual_day_eligible separately identifies eligible register outcomes. Local applied outputs are recorded without independent continuous lamp feedback. Regression uses the recorded FZ12, P8, P12 and P24 labels directly.

Separate numerical analyses
Model_Study contains optical parameter comparisons, common-input policy replays and sensitivity calculations. Its numerical units and assumptions are described in its README and the model supplement.
python run_study.py summarise-model --destination ../model_summary_output
To reproduce the full numerical project, install Model_Study/requirements_all.txt and run:
python run_study.py model-reproduce --workers 4 --destination ../model_analysis_output
These calculations are separate from the primary observation entry point.

Data navigation
Observed_Study/results: primary spatial, metered and diagnostic results.
Observed_Study/data/Figure_01.png and Figure_01.tif: the shared hardware figure.
Observed_Study/figures: PNG previews of the six plotted main figures. The drawing command regenerates EPS, SVG, PDF and PNG exports in a fresh destination.
Model_Study/mechanism_results: controlled numerical summaries.
documentation/publication_output_map.csv: sources for all 76 publication figures and tables.
documentation/observation_input_origin_map.csv: field-level archive extraction paths and hashes. Shared input files have one stored copy. Model_Study/Study/input_storage_map.json maps numerical input names to those copies; the numerical runner stages them into an isolated working directory. Storage location is distinct from analytical role.
documentation/primary_data_dictionary.csv: result fields, units and roles.
documentation/code_inventory.csv: primary source dependencies and hashes.
validation: current analysis verification and file-consistency checks.
Input values and raw log identifiers are retained. Boot identifiers distinguish recorded log sessions; software and protocol identifiers are not acquisition dates.

Authors
Zhongshuai Li, Yue Song, Yuxin Yan, Zhenhua Fan, Nanjie Zhou, Longhang Zhou, Zhejie Meng.
School of Energy and Power Engineering, Changchun Institute of Technology, Changchun 130012, China.
Corresponding author: Yue Song; yue_song@live.cn; https://orcid.org/0009-0008-7104-5368.
Author email addresses
Zhongshuai Li: lzs80vip@163.com
Yue Song: yue_song@live.cn
Yuxin Yan: 13935878466@163.com
Zhenhua Fan: Fanzh_h@163.com
Nanjie Zhou: Znj20038224830@163.com
Longhang Zhou: 18136486609@163.com
Zhejie Meng: 19862558422@163.com

Funding
This work was supported by Liaoyuan Power Supply Company, State Grid Jilin Electric Power Co., Ltd. [grant number 220240185].

Repository
https://github.com/thats0n9/four-circuit-lighting-control
Software: MIT; data: CC BY 4.0. Rights holders: Zhongshuai Li and Yue Song. See LICENSE, LICENSE_DATA.txt and LICENSING_STATUS.txt for the terms and scope.

Dictionary fields
CSV fields are column names and NPZ fields are array keys. Nested JSON fields use paths such as $[*]["lamp_coefficients"]. Wildcards select list records. Each path is verified against its source. From Repository, reproduce the dictionary with python documentation/build_data_dictionary.py --destination ../primary_data_dictionary.csv. Example outputs are siblings of Repository and must not already exist. The daylight_reference_lxh field is an illuminance-time integral in lx h.

Git storage
.gitattributes disables line-ending conversion for the archived files. Preserve this file when uploading or cloning so SHA256 checks remain valid. Development-environment folders, including .idea and .vscode, are excluded by .gitignore.
When updating an existing Git checkout, run git add --renormalize . from Repository after replacing the complete files, then git add .gitattributes .gitignore. This reapplies the byte-preservation rule to already tracked files. Remove tracked development-environment files before committing and pushing. Do not repair a transport mismatch by changing only the manifests.

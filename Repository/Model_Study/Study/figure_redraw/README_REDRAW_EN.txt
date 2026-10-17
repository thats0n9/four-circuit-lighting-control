Offline figure redraw toolkit | BSERT-CODE-20261017-C22-MODEL

Read ../README_DATA_CODE_EN.txt for the analysis roles of these data and figures.
redraw_all.py reads the exact plotting tables in data/ and the common figure_style.json. It does not run the study or import the control-analysis modules. Install requirements.txt and Times New Roman before drawing.

python redraw_all.py --output ../../../../model_figure_output
python redraw_all.py --figure 3 --output ../../../../single_model_figure_output
Run these direct examples from the figure_redraw directory. Each output directory must be new and outside Study; existing exports and archived previews are never overwritten.
The --figure order is Figure 1 to Figure 7 followed by Figures S1 to S4. On another operating system, supply a Times New Roman folder with --font-dir PATH. Font files are not distributed in this toolkit.

Every figure is 174 mm wide. Default axis text is 9 pt, ticks and legends 8.5 pt, titles 9.5 pt, lines 1.15 pt and markers 4.0 pt. All legends occupy a reserved area outside the data axes. figure_series_styles.csv defines the series colours and symbols; figure_data_index.csv maps figure panels to plotting tables.
SVG retains editable text. EPS and PDF embed TrueType fonts. PNG is exported at 800 dpi. Use vector files for layout editing, or import the CSV tables to recreate charts in Origin or another plotting application.
Keep data precision and the case/layout/point-count order. Figure 7 uses explicit lower and upper conditional bounds, not symmetric error bars around its nominal point. Figure S1 is a 36-by-12 acceptance matrix ordered depth/area/window and 8/12/16/24 points. The original photograph is retained byte for byte; Figure 1 displays a crop with numbered component callouts. No photographic content is reconstructed.

Supported model command: python run_study.py draw --destination ../../model_figure_output, run from the Model_Study directory. Supply --font-dir on systems where Times New Roman is not registered. The command at the enclosing Repository root draws the main manuscript figures instead.

Panel specifications
Figure 2 uses five geometric panels at equal spatial scale; window sides and task-zone boundaries are labelled. Figure 5 uses an overview, a near-zero enlargement and a screening summary. One symbol represents each exact coordinate and acceptance class; the 26 coincident zero changes comprise 23 accepted and 3 failed conditions. No coordinates are jittered. Figure 6 retains absolute values; its quality-axis span is 0.2 pp and the within-policy range annotation is rounded to 3 decimals. Figure 7 retains full ranges in (a) and repeats them in a −3 to 3 pp detail in (b); arrowheads denote continuation, not new endpoints. Failed research acceptance is encoded in red crosses and dashed ranges. These are conditional outer bounds, not confidence intervals.

Editable SVGs and matching EPS/PDF files are generated in the new drawing destination, not stored repeatedly in the repository. The PNG previews for Figures M2–M7 and S1–S4 are in ../figures_for_submission_en/. Figure M1 shares ../../../Observed_Study/data/Figure_01.png. The original sixteen plotting CSVs are retained.

Label and symbol audit
All visible text pairs are checked for bounding-box intersections. In-plot annotations are checked against scatter-marker extents and explicitly registered guide lines. Manual print-size inspection complements these checks. Exact coincident observations are retained. Acceptance uses open teal circles in Figures 5 and 7; failed acceptance uses red crosses. J has an upright auto subscript and italic mathematical letter; W is italic. All mathematical glyphs use Times New Roman. Daily energy is labelled kWh/day. Figure 5 shows signed ±0.05 kWh/day and ±0.2 pp scales. Figure 7 shows the −0.2 pp improvement and +0.2 pp non-inferiority scales only in its local detail. DZU uses an open square so nearby PU and DZU points remain distinguishable. Figure S3 uses a composite reference/sentinel symbol at their unchanged shared position; bar colours encode the full-position condition, independently of the sentinel-screen text.
Hatching of the detectable-at-sentinels bar identifies a failed sentinel screen; bar colour independently represents the full-position condition.

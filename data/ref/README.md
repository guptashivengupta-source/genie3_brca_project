# Reference files

`tf_list.txt` - 1,639 human transcription factors, Lambert et al., *Cell* 2018.
Downloaded from http://humantfs.ccbr.utoronto.ca/download/v_1.01/TF_names_v_1.01.txt

Used to flag `regulator_is_tf` on every edge and, with `--tf-only`, to restrict the
candidate regulators.

The composition marker sets live in `src/signatures.py` rather than here because they are
short and belong with the code that scores them. They are the standard markers for each
compartment: adipocyte (ADIPOQ, FABP4, PLIN1, CIDEC and related lipid-droplet genes),
stromal/fibroblast (collagens, FAP, POSTN, DCN, LUM), immune (PTPRC, CD3D/E, CD8A, MS4A1,
CD68, HLA-DRA) and proliferation (MKI67, TOP2A, AURKA, CCNB1). They are deliberately small
and specific; swap in ESTIMATE or a deconvolution if you want finer control.

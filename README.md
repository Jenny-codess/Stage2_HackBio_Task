# Stage2_HackBio_Task
This is the stage 2 task from HackBio internship in AI for Genomics.

## Predicting Drug Sensitivity in Cancer (GDSC)

Using the same dataset as last week, build a machine learning models to predict the sensitivityof cancer cell lines to treatments using the ln(IC50) column.

There are two possible ways to approach this task. Either approach it as a classification task or regression task. If you are approaching it as a regression task, simply use the column LN_IC50 column.

If you are approaching it as a classification task i.e. to predict resistant and sensitive readings, we will perform a quantile-based split. Basically, anything below 25th percentile is sensitive and anything above 75% is resistant. Everything in between is discarded. You can adjust the thresholds if you like. See code below

lower = df["ln_IC50"].quantile(0.25) #0.25 can be adjust to any number between 0 and 1

upper = df["ln_IC50"].quantile(0.75) #0.75 can be adjust to any number between 0 and 1

Modeling: We suggest you try either random forest or xgboosting. They both have regression and classification methods.

Feature Importance and Evaluation: These are key components of your report and our grading. Justify everything like a biologist, don’t just look for numbers that pass a threshold.

#Findings and Discussion

1. Drug modeled, and why
I modeled Oxaliplatin (n = 1,287 cell-line measurements, one of the largest of the 246 drugs in this GDSC panel — second only to Ulixertinib at 1,302). Sample size mattered practically: after the 25th/75th percentile split this left 322 sensitive and 322 resistant cell lines (644 total), enough for a reasonably stable train/test split.
Biologically, oxaliplatin is a third-generation platinum compound that forms platinum–DNA adducts and interstrand cross-links, blocking replication and transcription and triggering apoptosis. It's first-line standard of care for colorectal cancer, and — unlike cisplatin — its adducts are less efficiently recognized by the mismatch repair (MMR) pathway. That gives a concrete, testable hypothesis going in: cell lines with defective MMR (MSI-high) should respond differently than MMR-intact (MSS/MSI-low) lines. This makes oxaliplatin a good choice not just for sample size but because it has a known, literature-grounded mechanism to check the model's feature importances against.

2. Model comparison — Random Forest vs XGBoost
Metrics              Random Forest        XG Boost
Accuracy                0.82                0.77
F1 (weighted)           0.82                0.77
ROC-AUC                 0.895               0.859

Random Forest outperformed XGBoost on every metric, by roughly 5 points of accuracy/F1 and 3.6 points of AUC. With a test set of only 194 cell lines, that accuracy gap corresponds to about 10 additional correctly-classified cases — a real but not enormous difference given the sample size, and it wasn't checked with cross-validation, so I'd treat "RF is better" as a working conclusion rather than a settled one.
A plausible reason for the gap: after one-hot encoding, the feature matrix is 644 rows × 592 columns — the number of features is almost as large as the number of samples, and a large share of those columns come from encoding CELL_LINE_NAME (essentially a per-sample ID). Random Forest's bagging and per-split feature subsampling is generally more robust to this kind of wide, sparse, low-signal-per-column data than gradient boosting, which can start fitting noise when there are more directions to split on than there is real signal. XGBoost also wasn't given class-balancing (class_weight='balanced' was only applied to the Random Forest), which could account for some of the gap independent of the algorithm itself.

3. Top features and biological interpretation
The top predictors for both models were dominated by a single underlying signal — lineage: blood cancer vs. solid tumor — showing up under several different encodings:

Growth Properties_suspension (by far the top feature in both models): cells grown in suspension rather than adherently are, in this dataset, almost exclusively hematological cancers (leukemias, lymphomas, myeloma). This column is effectively acting as a proxy for "is this a blood cancer line," and its dominance suggests oxaliplatin sensitivity separates strongly along that axis — consistent with literature showing hematological lines often have distinct DNA-damage-response and apoptotic thresholds compared with solid-tumor lines.
TCGA_DESC_all, TCGA_DESC_laml, GDSC Tissue descriptor 1_leukemia, GDSC Tissue descriptor 1_lymphoma, TCGA_DESC_dlbc: these all point at the same signal from different angles (ALL = acute lymphoblastic leukemia, LAML = acute myeloid leukemia, DLBC = diffuse large B-cell lymphoma), reinforcing that hematologic identity is the model's main axis of separation.
Cancer Type_nb / tissue descriptor 2 = neuroblastoma: neuroblastoma is a pediatric solid tumor with relatively intact apoptotic machinery, and platinum-class agents are clinically used in neuroblastoma regimens — biologically plausible that it clusters toward the sensitive class.
TCGA_DESC_coread (colorectal): this is the most clinically direct hit — oxaliplatin's actual first-line indication is colorectal cancer, so its appearance here is a good sanity check that the model is picking up real pharmacology rather than noise.
Microsatellite instability Status (MSI)_mss/msi-l: this is the most mechanistically interesting feature. MMR-intact (MSS/MSI-low) status showing up as predictive lines up directly with the biology described above — MMR-deficient (MSI-high) cells are known to be less able to trigger apoptosis in response to certain platinum-DNA lesions, so MSI status modulating predicted sensitivity is consistent with established pharmacogenomics, not just a statistical artifact.
TCGA_DESC_stad (gastric), TCGA_DESC_sclc (small-cell lung): both are tissue types where platinum-based chemotherapy is standard clinical practice, again consistent with known indications.
A caution: the Random Forest top-15 also included individual cell-line names (CELL_LINE_NAME_a549, CELL_LINE_NAME_lu-165, CELL_LINE_NAME_kyse-220). A specific cell line showing up as an "important feature" is a red flag — it means the model is partly memorizing which exact cell lines were sensitive/resistant rather than learning transferable tissue-level rules, which ties directly into the limitation below.

4. Limitations
Class sizes and test set size. The quantile split leaves 322 vs. 322 lines, discarding the middle 50% of the dose-response range — the "ambiguous" intermediate responders that are arguably the most clinically realistic cases. With only 194 test samples, each misclassified case moves accuracy by ~0.5%, so reported metrics carry real sampling noise.
CELL_LINE_NAME inflates and contaminates the feature space. One-hot encoding it produces most of the 592 total feature columns against only 644 rows (features ≈ samples), and it's visibly leaking into the Random Forest's top predictors as individual cell-line identities rather than generalizable biology. This should be dropped from cat_cols in a revision.
Metadata, not genomics. CNA, Gene Expression, and Methylation in this file are Y/N flags for whether that omics data exists for a cell line, not the actual expression/mutation values — so this model is really learning from coarse tissue and clinical-metadata categories, not fine-grained molecular biology.
Single-drug scope. This model is specific to oxaliplatin's mechanism (DNA cross-linking). The tissue/lineage pattern found here is plausible for other platinum or DNA-damaging agents but wouldn't be expected to transfer to, say, a targeted kinase inhibitor.
No cross-validation or hyperparameter tuning. Metrics come from a single train/test split with fixed hyperparameters; a k-fold CV would give a more reliable estimate of the RF vs. XGBoost gap before concluding one model is definitively better.

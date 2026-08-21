# Paper t-SNE figure plan

This folder is intentionally independent of the training and evaluation code. It
only reads artifacts already written below an experiment directory:

- `predictions.csv`
- `recovered_features_records/index.csv`
- `recovered_features_records/<corruption>/*.pt`

The plotting code never imports or mutates the model, optimizer, dataloader, or
experiment configuration.

## Figure 1 — discriminative features under single-modality corruption

- **Question:** Are the adapted fused features class-discriminative when one
  modality is corrupted?
- **Preferred comparison:** Source versus AdaPGC, using the same corruption,
  same sample IDs, same selected classes, and a single joint PCA+t-SNE fit.
- **Default selection:** Five classes with the largest *minimum* common sample
  count across all compared conditions. This is support-based rather than
  performance-based selection. Samples are balanced per class.
- **Visual:** One panel per method, identical axes, class encoded by an explicit
  color and marker map. No more than five classes and 150 points per class.
- **Quantitative guardrail:** k-NN balanced accuracy, silhouette score, and
  Fisher ratio are computed in the shared PCA feature space, never from t-SNE
  coordinates.
- **Fallback:** If Source features were not saved, plot AdaPGC only and describe
  the figure as evidence of discriminative structure, not improvement.

Recommended main-paper conditions: one representative audio corruption and one
representative video corruption at the same severity. `gaussian_noise` is a
reasonable modality-symmetric choice when it exists in both streams; otherwise
use one canonical corruption from each stream and state the choice explicitly.

## Figure 2 — recovery fidelity under single-modality missingness

- **Question:** For the same partially missing samples, how are the available
  representation, posterior-mean recovery, and clean missing-modality ground
  truth arranged in a common feature geometry?
- **Visualized recovery summary:**
  `z_posterior_mean = sum_c alpha[c] * cond_means[c]`. This is the posterior
  mean of the saved class-conditional recovered fused features. The classifier
  marginalizes class-conditional GDA scores rather than feeding this one mean
  vector forward, so the figure must call it a posterior-mean recovery, not an
  internal feature used verbatim by the classifier. The script does **not** use
  the ground-truth label to select a conditional mean.
- **Available:** The missing run's `feat` selected by `audio_only` or
  `video_only` mask.
- **Reference:** Clean `ca` for missing audio or clean `cv` for missing video,
  matched by the exact same `sample_name`.
- **Default selection:** Four classes with the largest number of exact
  available/recovered/missing-GT triplets; balanced sampling within class.
- **Visual:** One joint t-SNE panel. Representation is encoded by explicit
  color/fill and class by marker shape. Only a small deterministic subset of
  recovery-to-missing-GT pair lines is drawn.
- **Quantitative guardrail:** paired cosine similarity, relative L2 error,
  reference-centroid agreement, and class-centroid displacement are computed
  before t-SNE for all three pairwise representation comparisons.
- **Exclusion:** Warm-up fallback records have no conditional recovered feature
  and are reported but never silently plotted as recovered features.

Recommended main-paper conditions: one `missing_a_*` and one `missing_v_*`
condition at the same missing ratio. The default recommendation is 0.70 when it
was collected; generate one overlay for each missing direction rather than every
missing ratio.

**Semantic caveat:** `cond_means` predicts fused feature space `F`; clean
missing-modality ground truth is modality-specific `ca`/`cv`. This plot supports
a feature-geometry/alignment statement, not literal reconstruction of a saved
modality-specific vector.

## Supplementary Figure S1 — modality-to-fused geometry

- **Question:** Do audio-only, video-only, and fused clean representations retain
  compatible class structure?
- **Data:** `ca`, `cv`, and the full subset of `feat` from the same clean forward
  records.
- **Visual:** Three facets in one joint embedding, at most four classes.
- **Use:** Diagnostic/supporting evidence only; it should not replace the direct
  missing-recovery comparison.

## Supplementary Figure S2 — missing-ratio centroid trajectory

Use only three predeclared missing ratios (for example 0.50, 0.70, 0.90) and at
most three classes. Plot class centroids and uncertainty ellipses rather than all
sample points. This avoids a dense condition-by-class legend. It is worth adding
only if exact common sample IDs exist across the selected ratios; otherwise use
the quantitative recovery metrics versus missing ratio instead of t-SNE.

## Interpretation rules

1. t-SNE is qualitative. Never claim distribution equality or improved
   separation from t-SNE alone.
2. All compared methods/states must share one fitted embedding and identical
   axes. Independently fitted panels are not comparable.
3. Class selection, sample IDs, seed, effective perplexity, and source paths are
   written to the output manifest.
4. Main figures use PDF for the paper and 300-DPI PNG for inspection.
5. If sample names or labels disagree across experiment directories, stop with
   an actionable error instead of silently joining approximate records.
6. Exact clean/missing pairs are strongest when both runs start from the same
   checkpoint and use the same seed/order. If model adaptation trajectories
   differ, state that the comparison also contains run-state variation.

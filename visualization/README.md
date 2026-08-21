# AdaPGC paper feature visualizations

`plot_feature_tsne.py` reads the feature records already collected by this
repository and produces publication-ready PDF/PNG figures, point tables,
quantitative checks, and a reproducibility manifest. It does not run adaptation
or modify any experiment artifact.

## Inputs expected

Each experiment directory must contain:

```text
EXP_DIR/
  predictions.csv
  recovered_features_records/
    index.csv
    <corruption>/batch_*_forward_results.pt
    <corruption>/batch_*_x2f_a.pt   # when audio is observed
    <corruption>/batch_*_x2f_v.pt   # when video is observed
```

These are exactly the files written when the archived run used
`--save-recovered-features-records`. Run this script only on trusted `.pt`
artifacts because loading PyTorch files can execute pickle payloads.

Dependencies are the project's normal PyTorch stack plus:

```bash
pip install numpy matplotlib scikit-learn
```

## 1. Inspect before plotting

```bash
python visualization/plot_feature_tsne.py inspect \
  --exp /path/to/exp_logs/EXPERIMENT
```

This checks the index and reports available corruptions, record types, sources,
sample counts, and warm-up batches.

## 2. Single-modality corruption: Source versus AdaPGC

Use the same corruption and severity for both experiments. The script keeps only
exact matching `sample_name` values, verifies their labels, balances classes,
and fits one shared PCA+t-SNE embedding.

```bash
python visualization/plot_feature_tsne.py corruption \
  --condition Source=/path/to/source_exp \
  --condition AdaPGC=/path/to/adapgc_exp \
  --corruption gaussian_noise \
  --severity 5 \
  --label-csv /path/to/class_labels_indices_ks50.csv \
  --output-dir paper_figures/tsne \
  --name fig_corrupt_audio_gaussian
```

Recommended main figure: make one audio-corruption plot and one video-corruption
plot with the same five classes. After the first plot, copy the selected class
IDs from its manifest and pass them explicitly to the second plot:

```bash
  --classes 3 11 18 27 41
```

This prevents the two modality panels from silently using different classes.
If Source feature records do not exist, pass only the AdaPGC condition. In that
case the plot supports a claim of discriminative structure, not a claim that
AdaPGC improved over Source.

## 3. Single-modality missing: available, recovered, and missing ground truth

One joint t-SNE panel now overlays three representations for the exact same
missing samples:

1. **Available:** the missing experiment's `feat` selected by `audio_only` or
   `video_only` mask;
2. **Recovery:** the posterior mean derived from the missing experiment's
   `alpha` and `cond_means`;
3. **Missing ground truth:** the clean experiment's modality-specific `ca` or
   `cv` for the exact matched `sample_name`.

The recovery summary is:

```text
recovered_feature[b] = sum_c alpha[b,c] * cond_means[b,c,:]
```

It never uses the ground-truth class to choose a conditional feature. This mean
is a faithful summary of the saved recovery distribution, but it is not a claim
that the classifier directly consumes this vector: the archived implementation
applies GDA to every class-conditional mean and then marginalizes those scores.
Also note that `cond_means` lives in fused feature space `F`, while the requested
missing ground truth is modality-specific `ca`/`cv`. The joint plot is therefore
a geometry/alignment visualization; do not describe it as direct reconstruction
of a modality-specific vector.

Audio missing, video observed:

```bash
python visualization/plot_feature_tsne.py recovery \
  --clean-exp /path/to/clean_exp \
  --missing-exp /path/to/missing_exp \
  --missing-corruption missing_a_0.70 \
  --label-csv /path/to/class_labels_indices_ks50.csv \
  --output-dir paper_figures/tsne \
  --name fig_recovery_missing_audio
```

Video missing, audio observed:

```bash
python visualization/plot_feature_tsne.py recovery \
  --clean-exp /path/to/clean_exp \
  --missing-exp /path/to/missing_exp \
  --missing-corruption missing_v_0.70 \
  --label-csv /path/to/class_labels_indices_ks50.csv \
  --output-dir paper_figures/tsne \
  --name fig_recovery_missing_video
```

The observed source (`v` for `missing_a_*`, `a` for `missing_v_*`) is inferred.
Pass `--source a` or `--source v` only for nonstandard condition names.

Representation type is encoded by color/fill, while class is encoded by marker
shape. By default six faint recovery-to-missing-GT pair lines per class are
drawn. Disable them with:

```bash
--overlay-pairs-per-class 0
```

Warm-up fallback batches do not contain `cond_means`; the script excludes them
and writes the excluded count to the manifest. It fails rather than substituting
classification logits for a feature vector.

## 4. Supplementary clean modality alignment

```bash
python visualization/plot_feature_tsne.py alignment \
  --exp /path/to/clean_exp \
  --corruption clean \
  --label-csv /path/to/class_labels_indices_ks50.csv \
  --output-dir paper_figures/tsne
```

This puts `ca`, `cv`, and full fused `feat` into one embedding. Keep it as a
supplementary diagnostic; the exact available/recovered/missing-GT triplets are
the direct visualization for the missing-modality analysis.

## Output contract

Every plotting command writes:

- `<name>.pdf`: vector paper figure;
- `<name>.png`: 300-DPI inspection copy;
- `<name>_points.csv`: exact plotted t-SNE coordinates and sample IDs;
- `<name>_metrics.csv`: metrics computed outside t-SNE;
- `<name>_manifest.json`: paths, class/sample selection, seed, effective
  perplexity, PCA settings, and recovery exclusions.

Defaults deliberately limit the view to five corruption classes or four
recovery classes, with at most 120 samples per class. Override only when the
rendered legend and point density remain readable.

## Important validity checks

- Do not compare independently fitted t-SNE panels. This script jointly fits all
  states shown in a figure and shares axes.
- Do not select classes by the best visual separation. Automatic selection uses
  only common sample support; explicit choices are recorded in the manifest.
- Do not interpret t-SNE distance as recovery error. The metrics CSV contains
  recovery-to-missing-GT, available-to-missing-GT, and recovery-to-available
  paired comparisons computed before t-SNE.
- If one experiment contains several severities, pass `--severity`. The current
  record filenames do not encode severity and can be overwritten by later
  severities; the script warns about this ambiguity.
- For the two recovery directions, use the same explicit class IDs whenever
  enough exact pairs exist.
- Keep the clean and missing runs on the same checkpoint, seed, and data order
  when possible. If their online adaptation trajectories differ, disclose that
  the feature comparison includes this run-state variation.

The rationale and recommended main/supplementary figure set are documented in
`FIGURE_PLAN.md`.

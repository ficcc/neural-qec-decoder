# Run records

One folder per recorded run: `manifest.json` (environment, code version, configurations, test results, notes) and `summary.json` (every key metric), plus a copy of the result files. Compare two runs with:

```bash
python -m scripts.compare_runs runs/A runs/B --min-rel 0.05
```

The commit hashes stored inside the manifests refer to the development history as it was when each run was recorded, before the history was tidied for publication. Commit titles and dates in the published history identify the same states.

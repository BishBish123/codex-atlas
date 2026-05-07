# LLM-as-judge calibration report

> **WARNING — PLACEHOLDER DATA**  
> The bundled `evals/calibration.csv` contains stub human labels.  
> This kappa value is **meaningless** until real human annotations replace  
> the `human_score` column.  See the calibration workflow in INTERPRETATION.md.

## Summary

- Judge mode: `auto`
- Samples: 10
- Agreed: 10 / 10
- Disagreed: 0 / 10
- **Cohen's kappa: 1.0000**

### Agreement matrix

| | Judge=1 | Judge=0 |
| --- | ---: | ---: |
| Human=1 | 9 | 0 |
| Human=0 | 0 | 1 |

### Divergence cases

| qid | human | judge |
| --- | --- | --- |
_No divergence cases._

### Kappa interpretation (Landis & Koch 1977)

| Range | Label |
| --- | --- |
| < 0 | Less than chance |
| 0.00-0.20 | Slight |
| 0.21-0.40 | Fair |
| 0.41-0.60 | Moderate |
| 0.61-0.80 | Substantial |
| 0.81-1.00 | Almost perfect |

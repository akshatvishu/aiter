# MI300X per-tensor quantization verification

The current-base operator checks passed. All 96 eligible timing cases improved
in every paired round; none of the 96 fallback cases exceeded the regression
threshold. This report records the 2026-10-16 measurement, not a new GPU run.

## Source and scope

- AITER base: `6d2aa05b63030cc10fd5bc06329c85bd222bf931`.
- Measured native candidate: `50d572a196401f820e64df9a4b8a7d2514b7fa67`.
- Final tested source: `6dbad62f60cdffd6f2ba808214296b1870437a18`.
  That follow-up changes Python import order only; both correctness suites were
  rerun with the identical native libraries.
- GPU: MI300X, gfx942. Torch 2.13 git 733fca1, HIP 7.2.53211.
- Timing rows: 1, 21, 511, 512, 513, 4096.
- Timing widths: 3072, 4096, 8192, 12288.
- Inputs: FP16/BF16. Outputs: INT8 and E4M3 FNUZ FP8.

Rows 8192/16384 and widths 14336/28672 were not timed. No gfx950 run,
general LLM prefill claim, or new end-to-end model speedup is established.
The earlier 5.69% Qwen request gain is historical evidence on an older stack.
Only callers already using `dynamic_per_tensor_quant` inherit this change.

## Correctness and timing

Each native arm passed 168 raw-byte and scale-bit exact cases plus 32 dispatch
checks. Coverage includes nonfinite values, alignment, non-contiguous output
fallback, and widths not divisible by 16. Input, output and scale storage must
not overlap; the original restricted pointers do not establish aliasing support.
Output-layout checks preserve legacy flat writes, not general strided support.

Three independent A/A pairs and three alternating base/candidate pairs used
eleven samples per case. Warm timing used 100-call graphs. Flushed timing used
a one-call graph after a 256 MiB write outside the event interval. The frozen
threshold was the greater of 5% and the largest observed A/A variation.
Eligible median reductions ranged from 13.14% to 71.92%. Fallback median
changes ranged from -2.13% to +0.67%.

Representative BF16 to FP8 results at 4096 rows:

| Width | Flushed base, us | Flushed candidate, us | Paired latency reduction |
| --- | ---: | ---: | ---: |
| 3072 | 63.946 | 31.793 | 50.59% |
| 12288 | 103.555 | 85.795 | 17.15% |
| 4096 | 70.320 | 36.763 | 47.72% |
| 8192 | 82.628 | 55.366 | 32.96% |

Displayed times are medians of process medians. Percentages are medians of
paired reductions, so their ratio need not equal the displayed percentage.
Native compiler flags matched; the quant object appeared last in the baseline
link order and first in the candidate. The missing-tabulate initial attempt
was rerun after supplying that harness dependency. Final runs exited zero.

Evidence: [contract](contract.json), [every case](summary.csv),
[summary](summary.json), [samples and dispatch records](samples/), and
[hook audit](hook-audit.json). Samples preserve all numerical rows and source
and library hashes while omitting host paths and GPU UUID. Original record
hashes are included. Binary artifacts and full logs remain in the local archive.

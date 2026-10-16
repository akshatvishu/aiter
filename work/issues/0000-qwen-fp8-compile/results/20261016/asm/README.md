# gfx942 FP8 FMHA regression verification

The test-only update adds a `__main__` entry point. AITER's CI runner invokes
`python3 "$file"`; the previous test only defined functions and exited zero.
Direct-script CPU collection finds seventeen tests with this update and none
with the old file. The GPU harness separately invoked pytest explicitly.

## Tested result

- Previous PR head: `f8731b53c3fb5a88c9acee15d3e5f65e540f6ff3`.
- Verified test commit: `1bc8f6ae1f5295f0a6bf5b99237642fb9ba8b11d`.
- Original binary: 6 failures, 11 passes.
- Existing patched binary: 17 passes, no errors or skips.
- No binary changes in the verification commit.
- GPU scope: MI300X gfx942, B1, H24, D128, noncausal dense FP8,
  S4096 and S4117. Loaded code-object paths were checked.

The exact spread represents logits -7-code/64 without rounding them during
FP8 input conversion. First and last maxima, uniform scores and nonconstant V
are covered. Cases attempt five calls each; a failed assertion stops a case.

## Tolerances and remaining measurement

Uniform and nonconstant-V cases keep absolute tolerance 0.01. Nonuniform
constant-V cases use 0.05 rather than the old 0.01. Correlated FP8 rounding
persists after probability scaling; an earlier scale sweep found up to 2.24%
spread error at other scales. This tolerance is fixture-specific, not a
universal FP8 attention accuracy guarantee. It was fixed before the GPU run.

| Original first-maximum fixture | Recorded maximum absolute error |
| --- | ---: |
| -7, both lengths | 0.0546875 |
| -8, S4096 | 0.578125 |
| -8, S4117 | 0.580078125 |
| Exact spread, both lengths | 0.14453125 |

Patched per-case maximum errors were not recorded. Passing establishes the
asserted bounds, not the observed margins below them. A diagnostic replay must
record those values before a measured margin is quoted. Do not choose new
tolerances after looking at that replay merely to obtain passing tests.

The original -7 error is only slightly above 0.05; the -8 and spread cases
provide stronger separation. Grouped, causal, MI308 and other architecture
kernels remain outside the binary fix and this verification.

Evidence: [native outcomes and hashes](results.json),
[script entry check](direct-script-check.json), [hook audit](hook-audit.json).
The actual upstream hook, Black and Ruff preserved the ASCII test byte for
byte. Full logs and JUnit files remain in the local experiment archive.

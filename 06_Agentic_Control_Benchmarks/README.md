# Benchmarking agentic control and system identification

Defined dynamic-system examples and control tasks, ran experiments, and evaluated agent/model performance for team-provided MPC, adaptive-control and system-identification modules.

## My contribution

My contribution is the benchmark cases, experiment setup, testing, metrics, comparisons and reporting. The team supplied the core controller/identification modules; I do not claim sole authorship of those systems.

## Selection and status

Includes selected personal testing/reporting code from a public training repository and saved MPC, adaptive-control and system-identification plots. The public training suite uses mocked agents and is distinct from the real-model experiments.

## Scope of this sample

Core company/team agent implementations, prompts, API keys and private datasets are not included. The local driver reference and the team's private core are excluded from this public repository. Plot labels alone do not prove successful LLM calls: fallback runs must be distinguished using the run manifests. Derivative-validation MSE and state-trajectory verification MSE are different metrics; cross-system raw errors have different physical units. No new ranking or acceptance-rate claim is inferred here.

Open `media` for the existing figures/videos and `code` for selected source. The portfolio-wide `SOURCE_MANIFEST.json` records file hashes and extraction details. Full original research folders remain separate.


## Saved MPC and adaptive-control figures

Eight original figures are included alongside the two existing system-identification plots. The HTML portfolio groups them by module. Click any image there for the full resolution.

| File in `media/` | Content | Archived plot snapshot |
|---|---|---|
| `mpc_score6_dashboard.png` | MPC: saved comparison score, token usage and estimated run cost | 2026-10-18_12h |
| `mpc_score6_seed_variability.png` | MPC: score variation across seeds (quartiles, median, mean and range) | 2026-10-18_12h |
| `mpc_two_link_manipulator_states.png` | MPC: two-link manipulator state regulation — GPT-4.1 Nano, seed 42 | 2026-10-18_12h |
| `mpc_two_link_manipulator_inputs.png` | MPC: corresponding joint control inputs and bounds — GPT-4.1 Nano, seed 42 | 2026-10-18_12h |
| `adaptive_score6_dashboard.png` | Adaptive: saved comparison score, token usage and estimated run cost | 2026-10-17_17h |
| `adaptive_score6_seed_variability.png` | Adaptive: score means and sample standard deviations across seeds | 2026-10-17_17h |
| `adaptive_two_link_manipulator_states.png` | Adaptive: two-link manipulator state regulation — GPT-4.1 Nano, seed 42 | 2026-10-17_17h |
| `adaptive_two_link_manipulator_inputs.png` | Adaptive: corresponding joint control inputs — GPT-4.1 Nano, seed 42 | 2026-10-17_17h |

The score-6 dashboards show a continuous benchmark score, not a count of passed trials. Seed displays retain their original definitions: MPC uses quartiles/median/mean/range; adaptive uses mean and sample standard deviation. Cost figures are historical local price-table estimates. The robot examples are state-regulation tests for the two-link manipulator, GPT-4.1 Nano, seed 42; they are not newly executed experiments. Source directories and hashes are recorded in SOURCE_MANIFEST.json.

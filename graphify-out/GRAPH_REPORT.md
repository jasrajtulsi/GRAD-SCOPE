# Graph Report - .  (2026-07-04)

## Corpus Check
- Large corpus: 1492 files · ~1,830,065 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 317 nodes · 508 edges · 14 communities (12 shown, 2 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 21 edges (avg confidence: 0.8)
- Token cost: 48,098 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Baseline Training Loop|Baseline Training Loop]]
- [[_COMMUNITY_Drift Detectors (GradNorm)|Drift Detectors (GradNorm)]]
- [[_COMMUNITY_Fragility Gate (Phase 3)|Fragility Gate (Phase 3)]]
- [[_COMMUNITY_MCP Analysis Logic|MCP Analysis Logic]]
- [[_COMMUNITY_Layer Health Classification|Layer Health Classification]]
- [[_COMMUNITY_Lead-Time Analysis|Lead-Time Analysis]]
- [[_COMMUNITY_Gradient Logger Hooks|Gradient Logger Hooks]]
- [[_COMMUNITY_Pipeline Concepts & Reports|Pipeline Concepts & Reports]]
- [[_COMMUNITY_ResNet-20 Architecture|ResNet-20 Architecture]]
- [[_COMMUNITY_Synthetic Drift Generator|Synthetic Drift Generator]]
- [[_COMMUNITY_MCP Server Tools|MCP Server Tools]]
- [[_COMMUNITY_VGG-11 Architecture|VGG-11 Architecture]]
- [[_COMMUNITY_Models Package Init|Models Package Init]]
- [[_COMMUNITY_MCP Server Doc|MCP Server Doc]]

## God Nodes (most connected - your core abstractions)
1. `GradientLogger` - 19 edges
2. `run_baseline()` - 17 edges
3. `StandIn` - 11 edges
4. `compute_all()` - 11 edges
5. `main()` - 10 edges
6. `BaselineConfig` - 9 edges
7. `WindowDetector` - 9 edges
8. `_load()` - 8 edges
9. `classify()` - 8 edges
10. `classify_all()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `GSNR signal and lead-time estimation` --semantically_similar_to--> `Windowed GradNorm production detector`  [INFERRED] [semantically similar]
  gradscope-mcp/README.md → grader-drift-pipeline/README.md
- `GSNR signal and lead-time estimation` --semantically_similar_to--> `GSNR gate metric (generalization-gap signal)`  [INFERRED] [semantically similar]
  gradscope-mcp/README.md → grader-drift-pipeline/results/gate_calibration.md
- `BaselineConfig` --uses--> `GradientLogger`  [INFERRED]
  experiments/baseline.py → logger/gradient_logger.py
- `build_model()` --calls--> `get_resnet20()`  [EXTRACTED]
  experiments/baseline.py → models/resnet20.py
- `build_model()` --calls--> `get_vgg11()`  [EXTRACTED]
  experiments/baseline.py → models/vgg11.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Phase 0-3 failure-early-warning pipeline** — grader_drift_pipeline_readme_drift_generator, grader_drift_pipeline_readme_windowdetector, grader_drift_pipeline_readme_windowed_gradnorm, grader_drift_pipeline_readme_fragility_gate, grader_drift_pipeline_readme_standin [EXTRACTED 1.00]
- **Gradient-based early-warning signals** — readme_lead_time_analysis, grader_drift_pipeline_readme_windowed_gradnorm, grader_drift_pipeline_results_gate_calibration_gsnr_metric, gradscope_mcp_readme_gsnr_signal_lead_time [INFERRED 0.75]
- **Phase 3 dual-metric gate decision flow** — grader_drift_pipeline_readme_fragility_gate, grader_drift_pipeline_results_gate_calibration_dual_metric_rule, grader_drift_pipeline_results_gate_calibration_gsnr_metric [EXTRACTED 1.00]

## Communities (14 total, 2 thin omitted)

### Community 0 - "Baseline Training Loop"
Cohesion: 0.09
Nodes (35): DataLoader, device, BaselineConfig, build_dataloaders(), build_model(), evaluate(), get_device(), _layer_norms() (+27 more)

### Community 1 - "Drift Detectors (GradNorm)"
Cohesion: 0.10
Nodes (33): first_alarm(), gradnorm(), gradnorm_lastblock(), msp(), pred_entropy(), Module, ndarray, Tensor (+25 more)

### Community 2 - "Fragility Gate (Phase 3)"
Cohesion: 0.09
Nodes (32): append_history(), _demo(), finetune_and_measure(), gate(), load_config(), per_sample_stats(), ndarray, Phase 3: supervised fragility gate for candidate models.  Before a retrained can (+24 more)

### Community 3 - "MCP Analysis Logic"
Cohesion: 0.10
Nodes (36): classify_gsnr(), _detect_failure(), _failure_epoch(), get_all_results(), get_current_gsnr(), get_layer_states(), get_lead_time_estimate(), get_training_status() (+28 more)

### Community 4 - "Layer Health Classification"
Cohesion: 0.10
Nodes (24): Enum, GRAD-SCOPE gradient-instrumentation package.  Exposes the gradient logger, layer, classify(), classify_all(), _FakeLogger, get_state_heatmap_data(), GradientState, _gsnr_values() (+16 more)

### Community 5 - "Lead-Time Analysis"
Cohesion: 0.12
Nodes (21): Any, compute_all(), detect_failure(), detect_signal_gsnr(), detect_signal_loss(), detect_signal_norm(), _iter_states(), _lead_time() (+13 more)

### Community 6 - "Gradient Logger Hooks"
Cohesion: 0.11
Nodes (11): GradientLogger, Module, Per-layer gradient-flow logger for GRAD-SCOPE.  Registers a full backward hook o, Flush the current epoch's state to ``live_state.json`` and advance.          Cal, Return the latest GSNR per layer (from the most recent backward pass)., Return the (epoch, GSNR) history for a single layer, in order., Write all collected per-(epoch, layer) records to a CSV file., Remove all registered backward hooks. (+3 more)

### Community 7 - "Pipeline Concepts & Reports"
Cohesion: 0.13
Nodes (21): GSNR generalization-gap paper (arXiv 2001.07384), cnn-grader-doctor MCP server, Phase 0 synthetic slow-drift generator, Phase 3 supervised fragility gate, Batch-level detector NO-GO decision, GradNorm-not-GSNR detection ruling, grader-drift-pipeline (gradescope), Stand-in conveyor CNN (+13 more)

### Community 8 - "ResNet-20 Architecture"
Cohesion: 0.14
Nodes (13): BasicBlock, get_resnet20(), _norm(), Module, Sequential, Tensor, ResNet-20 (CIFAR-style) for GRAD-SCOPE gradient-flow experiments.  3 groups of 3, Build a ResNet-20. When ``remove_bn`` is True, every BatchNorm layer is     repl (+5 more)

### Community 9 - "Synthetic Drift Generator"
Cohesion: 0.21
Nodes (14): BasicTransform, generate(), main(), make_transform(), Path, Phase 0: synthetic slow-drift corruption generator.  Applies ImageNet-C-style co, Severity in [0,1] -> albumentations transform at that intensity, p=1., Ramp corruption over `steps` time steps; returns manifest path.      Each step d (+6 more)

### Community 10 - "MCP Server Tools"
Cohesion: 0.15
Nodes (11): get_all_results(), get_current_gsnr(), get_layer_states(), get_lead_time_estimate(), get_training_status(), GRAD-SCOPE MCP server.  A FastMCP server (stdio transport) that exposes the live, Return the current gradient signal-to-noise ratio (GSNR) per layer.      Reads t, Return each layer's current gradient-health state.      Maps ``layer_name -> one (+3 more)

### Community 11 - "VGG-11 Architecture"
Cohesion: 0.19
Nodes (8): get_vgg11(), Sequential, Tensor, VGG-11 (CIFAR-style) for GRAD-SCOPE gradient-flow experiments.  Standard VGG-11, VGG-11 convolutional network for 32x32 inputs., Pathological initialization: set every weight/bias to the constant 0.001., Build a VGG-11. When ``remove_bn`` is True, no BatchNorm layers are added     (t, VGG11

## Knowledge Gaps
- **5 isolated node(s):** `gradscope-mcp`, `Phase 0 synthetic slow-drift generator`, `cnn-grader-doctor MCP server`, `Phase 3 gate threshold calibration report`, `Layer health state classification`
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GradientLogger` connect `Gradient Logger Hooks` to `Baseline Training Loop`, `Layer Health Classification`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Why does `get_resnet20()` connect `ResNet-20 Architecture` to `Baseline Training Loop`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Why does `StandIn` connect `Fragility Gate (Phase 3)` to `Drift Detectors (GradNorm)`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `main()` (e.g. with `gradnorm()` and `WindowDetector`) actually correct?**
  _`main()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `GRAD-SCOPE experiment definitions and runners.`, `Baseline training experiment for GRAD-SCOPE.  Trains an instrumented model on CI`, `Hyperparameters and settings for the baseline run.` to the rest of the system?**
  _117 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Baseline Training Loop` be split into smaller, more focused modules?**
  _Cohesion score 0.08677098150782361 - nodes in this community are weakly interconnected._
- **Should `Drift Detectors (GradNorm)` be split into smaller, more focused modules?**
  _Cohesion score 0.09957325746799431 - nodes in this community are weakly interconnected._
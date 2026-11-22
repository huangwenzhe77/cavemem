# ALFWorld short-reminder gate

These aggregate reports accompany the [experiment review](../../docs/alfworld-gate-review.md). The evaluation uses the frozen Qwen2.5-7B actor, one short reminder at an eligible checkpoint, and a total budget of 7,956 tokens per task. The extension contains 258 new game instances across 113 templates. The pooled analysis also includes 167 earlier evaluation games. Unmatched games enter the full denominator, and paired confidence intervals resample by template.

- [Test report](test-report.json) contains the independent extension and pooled policy results.
- [Memory-only control](memory-control-report.json) compares the learned gate with a controller that sees only memory properties and the budget.
- [Proposal ablation](ablation-intervals.json) reports paired effects after removing proposal features from the predictor. The matching rule still checks the original action for conflict.
- [Budget audit](budget-audit.json) verifies that live token cutoffs reproduced the archived outcome and token counts on seven development branches.

# Forge Eval — Anthropic

```
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Model/Backend                                  Scr      Acc      Cmp    Eff   Wst    Spd     N    rel   arg   tsl   b2s   s3s   crt   srn   err   dgr rel_s arg_s tsl_s b2s_s s3s_s crt_s srn_s err_s dgr_s
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
claude-sonnet-4-6 AN/N [reforged]           100.0%   100.0%   100.0%   100%   0.1   6.5s    50    100   100   100   100   100   100   100   100   100   100   100   100   100   100   100   100   100   100
claude-opus-4-6 AN/N [reforged]             100.0%   100.0%   100.0%   100%   0.1   8.5s    50    100   100   100   100   100   100   100   100   100   100   100   100   100   100   100   100   100   100
claude-haiku-4-5-20251001 AN/N [reforged]    99.6%   100.0%    99.6%    96%   0.5   4.0s    50    100   100   100   100   100   100    96   100   100   100   100   100   100   100   100    96   100   100
claude-haiku-4-5-20251001 AN/N [bare+any]    88.9%   100.0%    88.9%   100%   0.0   2.7s    50    100   100   100   100   100   100   100     0   100   100   100   100   100   100   100   100     0   100
claude-sonnet-4-6 AN/N [bare+any]            88.9%   100.0%    88.9%   100%   0.0   5.4s    50    100   100   100   100   100   100   100     0   100   100   100   100   100   100   100   100     0   100
claude-opus-4-6 AN/N [bare+any]              88.9%   100.0%    88.9%   100%   0.0   7.1s    50    100   100   100   100   100   100   100     0   100   100   100   100   100   100   100   100     0   100
claude-opus-4-6 AN/N [bare]                  88.6%   100.0%    88.6%   100%   0.0   9.0s    50    100   100   100   100   100   100   100     0   100   100   100   100   100   100   100   100     0    94
claude-sonnet-4-6 AN/N [bare]                87.2%    99.9%    87.3%   100%   0.0   6.8s    50    100   100    84   100   100   100   100     0   100   100   100    88   100   100   100    98     0   100
claude-haiku-4-5-20251001 AN/N [bare]        43.8%   100.0%    43.8%   100%   0.0   3.6s    50      0    92    98     2   100   100     0     0     0     0    96   100     0   100   100     0     0     0
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

Scr=score(correct/total), Acc=accuracy(correct/total, excl validate errors), Cmp=completeness(completed/total), Eff=efficiency(ideal/actual calls), Wst=avg wasted calls, Spd=avg time(excl compaction)
rel=relevance_detection, arg=argument_fidelity, tsl=tool_selection, b2s=basic_2step, s3s=sequential_3step, crt=conditional_routing, srn=sequential_reasoning, err=error_recovery, dgr=data_gap_recovery, rel_s=relevance_detection_stateful, arg_s=argument_fidelity_stateful, tsl_s=tool_selection_stateful, b2s_s=basic_2step_stateful, s3s_s=sequential_3step_stateful, crt_s=conditional_routing_stateful, srn_s=sequential_reasoning_stateful, err_s=error_recovery_stateful, dgr_s=data_gap_recovery_stateful
Ablation: full=all guardrails, no_rescue=no rescue loop, no_nudge=no rescue/retry nudge, no_steps=no step enforcement, no_recovery=no error recovery, no_compact=no compaction, bare=all guardrails off

*Generated 2026-04-11 12:42*

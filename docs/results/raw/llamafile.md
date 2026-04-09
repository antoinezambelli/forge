# Forge Eval — Llamafile

```
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Model/Backend                                              Scr      Acc      Cmp    Eff   Wst    Spd     N    rel   arg   tsl   b2s   s3s   crt   srn   err   dgr rel_s arg_s tsl_s b2s_s s3s_s crt_s srn_s err_s dgr_s
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
mistral-nemo:12b-instruct-2407-q4_K_M LF/P [reforged]    82.6%    83.2%    99.2%    95%   0.4   4.2s    50    100    98   100   100   100    98    86    56    20   100    98    70   100   100    98    84    72     6
mistral-nemo:12b-instruct-2407-q4_K_M LF/P [bare]        69.4%    79.1%    87.8%   100%   0.0   3.6s    50    100    88    96    98   100    90    80     0    18   100    96    52    98   100    56    72     0     6
llama3.1:8b-instruct-q8_0 LF/P [reforged]                66.0%    69.8%    94.6%    84%   1.0   3.8s    50    100    98   100   100   100    92     6    22    38   100    98   100   100     0    84     6    18    26
mistral:7b-instruct-v0.3-q8_0 LF/P [reforged]            63.8%    65.8%    96.9%    90%   0.7   4.9s    50    100    98   100   100   100    34    86    14    26   100   100     0   100    80    54    24     8    24
llama3.1:8b-instruct-q4_K_M LF/P [reforged]              63.8%    66.3%    96.2%    87%   0.9   3.0s    50    100   100   100   100   100    80     8    20    10   100    98   100    98     0    92     6    28     8
mistral:7b-instruct-v0.3-q4_K_M LF/P [reforged]          59.1%    60.8%    97.2%    87%   0.7   3.9s    50    100   100   100   100   100    16    88    16    10   100    98     0   100    76    22    20     8    10
llama3.1:8b-instruct-q4_K_M LF/P [bare]                  48.7%    55.9%    87.0%    97%   0.4   2.5s    50     96    96    50   100    38    64     8     0    12    98    96    48   100     0    64     2     0     4
llama3.1:8b-instruct-q8_0 LF/P [bare]                    39.9%    45.6%    87.6%   100%   0.1   2.5s    50    100    82    10   100     6    52     4     0    24    98    78    12   100     0    30     0     0    22
mistral:7b-instruct-v0.3-q4_K_M LF/P [bare]              26.8%    30.1%    88.9%   100%   0.0   2.2s    50    100     4   100    42     0     6    72     0     0   100     4     0    40     0    10     4     0     0
mistral:7b-instruct-v0.3-q8_0 LF/P [bare]                24.0%    27.2%    88.3%   100%   0.0   2.6s    50    100     0   100     0     2    26    72     0     2   100     0     0     4     0    24     2     0     0
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

Scr=score(correct/total), Acc=accuracy(correct/total, excl validate errors), Cmp=completeness(completed/total), Eff=efficiency(ideal/actual calls), Wst=avg wasted calls, Spd=avg time(excl compaction)
rel=relevance_detection, arg=argument_fidelity, tsl=tool_selection, b2s=basic_2step, s3s=sequential_3step, crt=conditional_routing, srn=sequential_reasoning, err=error_recovery, dgr=data_gap_recovery, rel_s=relevance_detection_stateful, arg_s=argument_fidelity_stateful, tsl_s=tool_selection_stateful, b2s_s=basic_2step_stateful, s3s_s=sequential_3step_stateful, crt_s=conditional_routing_stateful, srn_s=sequential_reasoning_stateful, err_s=error_recovery_stateful, dgr_s=data_gap_recovery_stateful
Ablation: full=all guardrails, no_rescue=no rescue loop, no_nudge=no rescue/retry nudge, no_steps=no step enforcement, no_recovery=no error recovery, no_compact=no compaction, bare=all guardrails off

*Generated 2026-04-09 15:06*

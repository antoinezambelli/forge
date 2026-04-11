# Forge Eval — Ollama

```
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Model/Backend                                              Scr      Acc      Cmp    Eff   Wst    Spd     N    rel   arg   tsl   b2s   s3s   crt   srn   err   dgr rel_s arg_s tsl_s b2s_s s3s_s crt_s srn_s err_s dgr_s
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
gemma4:31b-it-q4_K_M OL/N [reforged]                    100.0%   100.0%   100.0%    93%   0.3  17.0s    50    100   100   100   100   100   100   100   100   100   100   100   100   100   100   100   100   100   100
qwen3.5:27b-q4_K_M OL/N [reforged]                      100.0%   100.0%   100.0%    91%   0.3  13.9s    50    100   100   100   100   100   100   100   100   100   100   100   100   100   100   100   100   100   100
qwen3.5:35b-a3b-q4_K_M OL/N [reforged]                   99.9%    99.9%   100.0%    87%   0.5   5.2s    50    100   100   100   100   100   100   100   100    98   100   100   100   100   100   100   100   100   100
qwen3:14b-q4_K_M OL/N [reforged]                         96.3%    96.3%   100.0%    85%   0.6  19.6s    50    100   100   100   100   100   100   100    86    82   100   100   100    98    98   100   100    92    78
ministral-3:14b-instruct-2512-q4_K_M OL/N [reforged]     96.1%    96.9%    99.2%    79%   0.9   4.5s    50    100   100   100   100   100    96   100    88    86   100   100   100   100   100    90   100    96    74
ministral-3:8b-instruct-2512-q8_0 OL/N [reforged]        94.6%    95.6%    98.9%    70%   1.3   9.1s    50    100    96   100   100   100    92    98    84    84   100   100   100   100   100    94   100    84    70
gemma4:e4b-it-q4_K_M OL/N [reforged]                     93.0%    93.0%   100.0%    93%   0.2   6.6s    50    100   100   100   100   100    72   100   100    98   100   100   100   100   100    12   100   100    92
qwen3:8b-q8_0 OL/N [reforged]                            91.0%    91.0%   100.0%    89%   0.5  18.7s    50    100   100   100   100   100   100   100    30    90   100   100   100   100   100   100    98    26    94
ministral-3:8b-instruct-2512-q4_K_M OL/N [reforged]      91.2%    94.7%    96.3%    68%   1.5   4.2s    50    100    96   100   100    98   100    98    62    76   100    88   100   100    96   100    98    60    70
gemma4:e4b-it-q8_0 OL/N [reforged]                       90.1%    90.1%   100.0%    93%   0.2   8.6s    50    100   100   100   100   100    30    94   100    96   100   100   100   100   100    10    98    98    96
qwen3:8b-q4_K_M OL/N [reforged]                          86.9%    86.9%   100.0%    84%   0.7  12.4s    50    100   100   100   100   100   100   100    22    62   100   100   100   100   100    96    98    16    70
gemma4:26b-a4b-it-q4_K_M OL/N [reforged]                 85.1%    85.1%   100.0%    91%   0.3   3.2s    50    100   100   100   100   100   100   100    28    42   100   100   100   100   100   100   100    24    38
gemma4:31b-it-q4_K_M OL/N [bare]                         84.3%    95.0%    88.8%   100%   0.0  14.4s    50    100   100   100   100   100   100   100     0   100    98   100   100   100   100    20   100     0   100
qwen3.5:27b-q4_K_M OL/N [bare]                           81.1%    95.1%    85.3%   100%   0.0  10.9s    50     82   100    82    96   100   100   100     0   100    84   100    94    98   100    24   100     0   100
gemma4:e4b-it-q4_K_M OL/N [bare]                         79.3%    91.4%    86.8%    97%   0.1   7.8s    50    100   100    82   100    96    74    94     0    98   100   100    90   100    96     2    96     0   100
gemma4:e4b-it-q8_0 OL/N [bare]                           74.9%    87.4%    85.7%    97%   0.1   9.6s    50    100   100    72   100   100    30    90     0    88   100   100    78   100   100     6    92     0    92
gemma4:26b-a4b-it-q4_K_M OL/N [bare]                     68.8%    88.6%    77.7%    96%   0.1   2.7s    50    100   100     0    98   100   100   100     0    38   100   100     0   100   100    86   100     0    16
qwen3:8b-q8_0 OL/N [bare]                                64.7%    84.2%    76.8%    95%   0.2  17.2s    50     90    94     0    40    88   100   100     0    68    86    94     0    62    88    92   100     0    62
qwen3.5:35b-a3b-q4_K_M OL/N [bare]                       64.2%    97.8%    65.7%    98%   0.1   5.5s    50     58   100     0    18   100   100   100     0    98    74   100     0    32   100    78   100     0    98
qwen3:14b-q4_K_M OL/N [bare]                             62.6%    91.1%    68.7%    98%   0.1  19.0s    50     90    90     0     4    86   100   100     8    86    92    94     4    36    88    64   100     0    84
qwen3:8b-q4_K_M OL/N [bare]                              56.3%    81.5%    69.1%    96%   0.1  10.5s    50     54    96     0     6    98    98    98     0    28    58    96     4    80    96    78    96     0    28
mistral-nemo:12b-instruct-2407-q4_K_M OL/N [reforged]    44.6%    62.3%    71.6%    46%   3.7   7.9s    50     22     4    64    98    40    22    40    28    66     8    24    50   100    50    40    44    36    66
ministral-3:14b-instruct-2512-q4_K_M OL/N [bare]         34.6%    99.0%    34.9%    98%   0.1   1.9s    50    100    44    56     0   100     4     0     0     6   100    46    50     0   100     6     0     0    10
llama3.1:8b-instruct-q8_0 OL/N [reforged]                19.6%    27.5%    71.2%    39%   4.4   5.8s    50     98     6    42    10    14     0     6    10     6    76     6    40    14    16     0     0     6     2
llama3.1:8b-instruct-q4_K_M OL/N [reforged]              18.8%    25.1%    74.9%    40%   4.4   4.4s    50     92    10    48    42    22     0     0     4     2    60    10    20    10     8     0     0    10     0
ministral-3:8b-instruct-2512-q4_K_M OL/N [bare]          17.0%    86.0%    19.8%    91%   0.4   4.1s    50      0     0     0     0    66    98     0     0     4     0     4     0     0    70    60     0     0     4
ministral-3:8b-instruct-2512-q8_0 OL/N [bare]            15.4%    81.8%    18.9%    93%   0.3   9.7s    50      0     0     2     2    52   100     0     0    10     0     0     8     0    44    58     0     0     2
mistral:7b-instruct-v0.3-q8_0 OL/N [bare]                 4.6%    15.5%    29.3%   100%   0.0   1.8s    50      0     0     0    82     0     0     0     0     0     0     0     0     0     0     0     0     0     0
mistral:7b-instruct-v0.3-q4_K_M OL/N [bare]               3.9%    12.3%    31.7%   100%   0.0   1.2s    50      0     0     0    70     0     0     0     0     0     0     0     0     0     0     0     0     0     0
mistral:7b-instruct-v0.3-q4_K_M OL/N [reforged]           1.1%    47.6%     2.3%    46%   3.5   3.9s    50      6     0     0     4     0     0     0     0     0    10     0     0     0     0     0     0     0     0
mistral-nemo:12b-instruct-2407-q4_K_M OL/N [bare]         1.3%    92.3%     1.4%   100%   0.0   0.6s    50     12     0     0     0     0     0     0     0     0    12     0     0     0     0     0     0     0     0
llama3.1:8b-instruct-q8_0 OL/N [bare]                     0.1%     0.4%    25.6%   100%   0.0   1.6s    50      0     0     0     0     0     0     0     2     0     0     0     0     0     0     0     0     0     0
llama3.1:8b-instruct-q4_K_M OL/N [bare]                   0.4%     2.2%    19.9%   100%   0.0   1.0s    50      0     0     0     0     0     0     0     8     0     0     0     0     0     0     0     0     0     0
mistral:7b-instruct-v0.3-q8_0 OL/N [reforged]             0.4%    30.8%     1.4%    33%   5.2   6.8s    50      6     0     0     0     0     0     0     0     0     2     0     0     0     0     0     0     0     0
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

Scr=score(correct/total), Acc=accuracy(correct/total, excl validate errors), Cmp=completeness(completed/total), Eff=efficiency(ideal/actual calls), Wst=avg wasted calls, Spd=avg time(excl compaction)
rel=relevance_detection, arg=argument_fidelity, tsl=tool_selection, b2s=basic_2step, s3s=sequential_3step, crt=conditional_routing, srn=sequential_reasoning, err=error_recovery, dgr=data_gap_recovery, rel_s=relevance_detection_stateful, arg_s=argument_fidelity_stateful, tsl_s=tool_selection_stateful, b2s_s=basic_2step_stateful, s3s_s=sequential_3step_stateful, crt_s=conditional_routing_stateful, srn_s=sequential_reasoning_stateful, err_s=error_recovery_stateful, dgr_s=data_gap_recovery_stateful
Ablation: full=all guardrails, no_rescue=no rescue loop, no_nudge=no rescue/retry nudge, no_steps=no step enforcement, no_recovery=no error recovery, no_compact=no compaction, bare=all guardrails off

*Generated 2026-04-11 12:42*

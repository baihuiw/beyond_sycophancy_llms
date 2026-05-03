# Participant Count Bug Documentation

## Summary

All experiments using country-specific conditions (India, Peru, USA) incorrectly used **175 participants** in prompts instead of the correct **58 participants**.

## Affected Experiments

- **Condition 3**: India with scale_1 (all variations 0-4)
- **Condition 4**: Peru with scale_1 (all variations 0-4)
- **Condition 5**: USA with scale_1 (all variations 0-4)

## Root Causes

### 1. Original Experiments (`final_experiments/`)

**Location**: `src/gpt_inference.py:74-81`

**Problem**: When `prompt_type=all`, the code hardcodes which prompt types to use:

```python
methods_to_generate = [
    PromptType.SD,           # Uses SD_PROMPT with 175
    PromptType.DISTRIBUTION,  # Uses DISTRIBUTION_PROMPT with 175
    PromptType.MEAN_ALONE,    # Uses MEAN_ALONE_PROMPT with 175
    PromptType.STD_ALONE,     # Uses STD_ALONE_PROMPT with 175
    PromptType.MEAN_REP,      # Uses MEAN_ALONE_PROMPT with 175
]
```

**What it should do**: For non-global countries, use:
- `PromptType.SD_C` (uses `SD_PROMPT_COUNTRY` with 58)
- `PromptType.DISTRIBUTION_C` (uses `DISTRIBUTION_PROMPT_COUNTRY` with 58)
- `PromptType.FIVE_HUNDRED_REP_PROMPT_C` (uses `FIVE_HUNDRED_REP_COUNTRY_PROMPT` with 58)

**Limitation**: Already in `hydra_refactor` branch countr-specific prompts for MEAN_ALONE and STD_ALONE defined in `src/distributional_alignment_prompt_builder.py` had no country templates. 


**What it should do**: Use conditional logic or separate templates:
- 175 for Global
- 58 for India, Peru, USA

## Correct Participant Counts (from `src/distributional_alignment_prompt_builder.py`)

- **Global**: 175 participants (lines 11, 21, 26, 32, 75)
- **India**: 58 participants (lines 16, 54, 79)
- **Peru**: 58 participants (lines 16, 54, 79)
- **USA**: 58 participants (lines 16, 54, 79)

## Prompt Template Coverage

| Method | Has Country Variant? | Line References |
|--------|---------------------|-----------------|
| SD (std) | YES - `SD_PROMPT_COUNTRY` | Lines 52-72 |
| DISTRIBUTION | YES - `DISTRIBUTION_PROMPT_COUNTRY` | Lines 79-82 |
| MEAN_ALONE | NO - only 175 version exists | Lines 20-23 |
| STD_ALONE | NO - only 175 version exists | Lines 25-28 |
| MEAN_REP | NO - uses MEAN_ALONE_PROMPT | Lines 20-23, 114 |

## Additional Bugs Found

### 3. Dilemma Ordering Bug

**Location**: `final_experiments_aggregated/generate_aggregated_batches.py:297`

**Problem**: Sorts questions alphabetically instead of numerically:
```python
df_var = df_var.sort_values("id").reset_index(drop=True)
```

**Result**: Produces order `moral_1, moral_10, moral_11, moral_2, moral_3, ...` instead of `moral_1, moral_2, ..., moral_11`

**Fix**: Extract numeric portion and sort:
```python
df_var['id_num'] = df_var['id'].str.extract('(\d+)').astype(int)
df_var = df_var.sort_values('id_num').drop('id_num', axis=1)
```

## Impact

All country-specific experiments told the LLM to estimate distributions for a sample of 175 participants when the actual human data had only 58 participants per country. This may have affected the model's calibration and estimates.

## Status

- Original experiments in `final_experiments/` were already run with incorrect prompts
- Batch files in `final_experiments_aggregated/` not yet copied to Llama (can be fixed before submission)

## Next Steps

1. Decide whether to re-run original experiments with corrected prompts
2. Fix `generate_aggregated_batches.py` before copying to Llama
3. Consider creating country-specific templates for MEAN_ALONE, STD_ALONE, MEAN_REP methods

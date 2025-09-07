# A LLM regressor to predict scores for candidates.
# A simple version: construct a prompt with the buffer statistics (candidate parameters and scores), ask LLM to analyze the pattern and predict scores for all candidates (candidates with noisy scores and new candidates without any statistics).
# Issues for the simple version:
# 1. Output limit is not enough: when the output limit of LLM is reached, the LLM with stop the generation, only part of the candidates have predicted scores. Asking LLM to continue the generation would solve this issue in some sense.
# 2. When we have a large buffer, the prompt is too long. LLM cannot do reasoning properly. It would output nonsenses, like repeating some numbers without any logic. 
# 3. LLM may skip some candidates due to reason 1 or 2.

# The key to fix these issues is to use less candidates in the prompt, and only predict part of the candidates that need predicted scores.
# When we need to predict scores for a batch of candidates, we first divide the batch into smaller batches, and predict scores for each smaller batch. 
# For each smaller batch, we sample a subset of candidates with statistics to construct the prompt, call LLM to make the prediction. To make the predicition more reliable, we repeat this process multiple times (with different subsets in the prompt) and take the average.
# Parallelize the process, then it only take the time of one LLM call the get the predicted scores for a small batch of candidates. Also parallelize the prediction for different batches.

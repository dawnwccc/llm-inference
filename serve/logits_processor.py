from transformers import LogitsProcessor, add_start_docstrings
import torch
from transformers.generation.logits_process import LOGITS_PROCESSOR_INPUTS_DOCSTRING


class InvalidScoreLogitsProcessor(LogitsProcessor):
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class FrequencyAndPresencePenaltyLogitsProcessor(LogitsProcessor):
    r"""
    The frequency and presence penalties found in the Chat completions API and Legacy Completions API can be used to
    reduce the likelihood of sampling repetitive sequences of tokens. They work by directly modifying the logits (
    un-normalized log-probabilities) with an additive contribution.

    mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence
    Where:

    mu[j] is the logits of the j-th token
    c[j] is how often that token was sampled prior to the current position
    float(c[j] > 0) is 1 if c[j] > 0 and 0 otherwise
    alpha_frequency is the frequency penalty coefficient
    alpha_presence is the presence penalty coefficient
    As we can see, the presence penalty is a one-off additive
    contribution that applies to all tokens that have been sampled at least once and the frequency penalty is a
    contribution that is proportional to how often a particular token has already been sampled.

    Reasonable values for the penalty coefficients are around 0.1 to 1 if the aim is to just reduce repetitive
    samples somewhat. If the aim is to strongly suppress repetition, then one can increase the coefficients up to 2,
    but this can noticeably degrade the quality of samples. Negative values can be used to increase the likelihood of
    repetition.
    """

    def __init__(self, frequency_penalty: float, presence_penalty: float):
        if not isinstance(frequency_penalty, float):
            raise ValueError(f"`frequency_penalty` has to be a strictly float, but is {frequency_penalty}")
        if not isinstance(presence_penalty, float):
            raise ValueError(f"`presence_penalty` has to be a strictly float, but is {presence_penalty}")
        # if not isinstance(vocab_size, int):
        #     raise ValueError(f"`vocab_size` has to be a strictly int, but is {vocab_size}")

        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        # self.sampling_record = torch.zeros()

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * self.frequency_penalty, score / self.frequency_penalty)

        scores.scatter_(1, input_ids, score)
        return scores

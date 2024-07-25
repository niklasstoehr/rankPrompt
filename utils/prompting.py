#prompting: prompting and constrained decoding functionality

import torch, operator

from utils import helpers
import numpy as np


def mlm_decode(texts, model, tokenizer, cand_toks=[], spaced=" ", norm=True):
    """
    constrained decoding with masked language model
    """

    ## (1) tokenize and embed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits

    ## (2) select mask token
    batch_idx, token_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)  # retrieve index of [MASK]
    mask_logits = logits[batch_idx, token_idx]

    ## (3) constrained decoding: get candidate token ids
    cand_toks = list(map(lambda tok: spaced + tok, cand_toks))  ## get spaced tokens
    cand_tok_ids = tokenizer(cand_toks).input_ids
    cand_tok_ids = list(map(lambda ids: ids[-2], cand_tok_ids))

    ## (4) select logits corresponding to tokens
    mask_logits = torch.index_select(mask_logits, -1, torch.tensor(cand_tok_ids).long().to(model.device))
    if mask_logits.shape[0] > 1 and norm:
        mask_logits = mask_logits - mask_logits.mean(0)  ## calibration by removing mean prediction
    cand_tok_logits, cand_tok_idx = torch.max(mask_logits, dim=-1)
    return cand_tok_idx.long().tolist(), cand_tok_logits.tolist()


def causal_decode(texts, model, tokenizer, cand_toks=[], spaced=" ", norm=True):
    """
    constrained decoding with causal (left-to-right) language model
    """

    noMask_texts = []
    if isinstance(texts, list) == False:
        texts = [texts]
    for text in texts:
        if "[MASK]" in text:
            noMask_texts.append(text.replace("[MASK]", '').rstrip())
        else:
            noMask_texts.append(text)

    ## (1) tokenize and embed
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(noMask_texts, padding=True, return_tensors="pt").to(model.device)

    ## generation output__________________
    gen_outputs = model.generate(
        inputs=inputs.input_ids,
        num_beams=1,
        min_length=None,
        max_length=None,
        max_new_tokens=10,
        output_scores=True,
        renormalize_logits=False,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id
    )

    ## (2) constrained decoding: score different candidate options by likelihood
    cand_scores = []

    for toks in cand_toks:

        ## (2.1) tokenize each candidate
        if isinstance(toks, list) == False: ## when tokenizing via tokenizer don't need "Ġ"
            toks = [toks]
        if len(spaced) > 0:
            toks = list(map(lambda tok: spaced + tok, toks))

        ## need tokenizer instead of convert because cands can be multiple tokens
        cand_ids = tokenizer(toks, return_tensors="pt").input_ids.to(model.device)
        gen_scores = operator.itemgetter(*list(range(0, cand_ids.shape[-1])))(gen_outputs.scores)  ## number of cands

        if isinstance(gen_scores, torch.Tensor):  ## if single token
            gen_scores = [gen_scores]  ## scores need to be list of pred_tokens x vocab_size

        ## (2.2) compute_transition_scores
        if gen_scores[0].shape[0] > 1:  ## batch in parallel
            cand_ids = cand_ids.repeat(gen_scores[0].shape[0], 1)

        gen_scores = model.compute_transition_scores(
            sequences=cand_ids,
            scores=gen_scores,
            normalize_logits=False
        )

        gen_scores = gen_scores.mean(-1)  ## normale by length
        cand_scores.append(gen_scores)

    cand_scores = torch.stack(cand_scores).view(-1, len(cand_toks))
    if cand_scores.shape[0] > 1 and norm:
        cand_scores = cand_scores - cand_scores.mean(0)  ## calibration by removing mean prediction

    cand_scores, cand_score_idx = torch.max(cand_scores, dim=-1)
    return cand_score_idx.long().tolist(), cand_scores.tolist()


def pointwise_ranking(texts, model, tokenizer, cand_toks, cand_ranks, cand_ids):
    """
    pointwise prompting and subsequent tie braking
    """
    if model.can_generate():  ## decoder-only models
        cand_idx, cand_score = causal_decode(texts, model, tokenizer, cand_toks=cand_toks, norm=False)
    else:  ## encoder-only models
        cand_idx, cand_score = mlm_decode(texts, model, tokenizer, cand_toks=cand_toks, norm=False)  ## returns list

    id_rank_score = list(zip(cand_ids, cand_idx, cand_score))
    id_rank_score.sort(key=lambda i: (i[1], i[2]), reverse=True)  ## sort by index, then scores
    rank_pred = list(map(lambda x: x[0], id_rank_score))
    #texts = operator.itemgetter(*rank_pred)(texts)
    return rank_pred, texts



def stepwise_decoding(text, model, tokenizer, cand_toks, cand_ranks=None, delim=", "):
    """
    stepwise (listwise) decoding
    """
    rank_pred = []
    n_tokens = len(cand_toks)
    for step in range(0, n_tokens):  ## single item per decoding step

        if step == 0:  ## at the beginning, add white space
            text = text + " "  ## white space

        if isinstance(model, str) == False:
            if model.can_generate() is False:
                masked_text = text + "[MASK]"  ## at [MASK] token for mlm models
            else:
                masked_text = text  ## add nothing, simply decode
        else:
            masked_text = text  ## add nothing, simply decode

        ## (1) constrained mask decoding
        if model.can_generate():  ## decoder-only models
            cand_idx, _ = causal_decode(masked_text, model, tokenizer, cand_toks=cand_toks)
            cand_idx = cand_idx[0]
        else:  ## encoder-only models
            cand_idx, _ = mlm_decode(masked_text, model, tokenizer, cand_toks=cand_toks)  ## returns list
            cand_idx = cand_idx[0]

        ## (2) build string recursively, record predictions and remove token from candidate lists
        text = f"{text}{cand_toks[cand_idx]}"
        cand_toks.pop(cand_idx)
        if cand_ranks is not None:
            rank_pred.append(cand_ranks[cand_idx])
            cand_ranks.pop(cand_idx)

        if delim is not None and step < n_tokens - 1:
            text += delim

    ## if some tokens have not been predicted, add missing ranks in random order
    if len(cand_ranks) > 1:
        random.shuffle(cand_ranks)
        # cand_ranks = [0] * len(cand_ranks)
        rank_pred = rank_pred + cand_ranks
    return rank_pred, text


## PROMPTING CLASS_________________________________________________

class Prompter():
    def __init__(self, model, tokenizer, x_y=None, print_prompt=False):
        self.tokenizer, self.model = tokenizer, model
        self.print_prompt = print_prompt
        # self.model.eval()
        self.x_y = x_y

        assert model.config.architectures[0] in ['PhiForCausalLM', 'GPT2LMHeadModel', 'DebertaForMaskedLM', 'MPTForCausalLM'], "need MLM with decoder or decoder-only model"
        print(f"{model.config.architectures[0]}")

    def forward_fn(self, batch, ent_idx=None, x_key="x_pred", y_key="y_pred"):

        ## extract batch____________________________________
        if self.x_y is None and "x_y" in batch.keys():
            self.x_y = batch["x_y"]

        ## get y
        if self.x_y[y_key] in batch.keys():
            y = batch[self.x_y[y_key]]
            if ent_idx is not None:
                y = operator.itemgetter(*ent_idx)(y)
        else:
            y = None

        ## get x
        texts = batch[self.x_y[x_key]]
        if ent_idx is not None:
            texts = texts[ent_idx]

        ## encoder only / pairwise___________________________________________

        if self.x_y["y_pred"] == "y":
            cand_toks = list(batch["tok"].keys())

            if self.model.can_generate():  ## decoder-only model
                y_pred_idx, y_pred_score = causal_decode(texts, self.model, self.tokenizer, cand_toks=cand_toks)
            else:  ## mlm model
                y_pred_idx, y_pred_score = mlm_decode(texts, self.model, self.tokenizer, cand_toks=cand_toks)

            pred_toks = []
            for cand_idx in y_pred_idx:
                pred_toks.append(cand_toks[cand_idx])

            if self.print_prompt:
                print("\n", list(zip(texts, pred_toks)), "\n")
            return (y_pred_idx, y_pred_score), y, None, None


        elif self.x_y["y_pred"] == "ranks":
            if "enums" in batch.keys():  ## listwise approach
                cand_toks, cand_ranks = list(batch["enums"].keys()), list(batch["enums"].values())
                rank_pred, pred_text = stepwise_decoding(batch["texts"], self.model, self.tokenizer,cand_toks=cand_toks, cand_ranks=cand_ranks)
            else:  ## pointwise approach
                rank_pred, pred_text = pointwise_ranking(batch["texts"], self.model, self.tokenizer,cand_toks=batch["toks"], cand_ranks=batch["ranks"],cand_ids=batch["ent_ids"])
            if self.print_prompt:
                if isinstance(pred_text, str):
                    print(f"\n{pred_text}, ranks: {rank_pred}\n")
                else:
                    print(f"\n{list(zip(pred_text, rank_pred))}\n")
            return None, None, rank_pred, y

    def predict(self, batch: dict, bound: int = 0.5, return_y: bool = False, **kwargs):

        y_pred, y_true, rank_pred, rank_true = self.forward_fn(batch)

        if y_pred is not None:
            ent_ids = batch["ent_ids"]  ## needed for mapping pairs to global ranking
            y_pred_idx, y_pred_score = y_pred[0], y_pred[1] ## unpack index and score
            rank_pred = helpers.pairwise_to_ranks(y_pred_idx, ent_ids, float_list=y_pred_score, tie=False)  ## even works if we don't know the ranks
            if return_y and y_true is not None:  ## for evaluation loop
                y_true_pair = list(zip(*y_true))[0]  ## y_true also works if we don't know the ranks
                rank_true = helpers.pairwise_to_ranks(y_true_pair, y_true, tie=False)
                return rank_pred, y_pred_idx, rank_true, y_true_pair
            return y_pred_idx

        elif rank_pred is not None:
            if return_y and rank_true is not None:  ## for evaluation loop
                y_pred_pair = helpers.get_pairings(rank_pred, pair_type="comb", n_tuple=2, get_argmax=True)
                y_true_pair = helpers.get_pairings(np.array(rank_true), pair_type="comb", n_tuple=2, get_argmax=True)
                return rank_pred, y_pred_pair, rank_true, y_true_pair
            return rank_pred  ## pair, rank
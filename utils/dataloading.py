#dataloading: loading data and formulating prompts
from abc import ABC, abstractmethod 
from string import Formatter
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

from utils import helpers

class Data_Loader(ABC):
    def __init__(self, prompt_type="point",**kwargs):
        self.prompt_type, self.kwargs = prompt_type, kwargs
        data = self.load(self.prompt_type)
        assert prompt_type == "list" or prompt_type=="pair" or prompt_type=="point_scale" or prompt_type=="point"
        data = prompt_postprocess(data, self.prompt_type, **kwargs)
        self.data = data
        print(f"prompt_type: {self.prompt_type}, kwargs: {kwargs}")

        @abstractmethod
        def load(self, data):
            raise NotImplementedError()


def prompt_postprocess(data, prompt_type, option_marker='"', add_examplar:bool=False):
    """
    method for postprocessing, particularly if model_type="prompt"
    """
    if add_examplar:
        exemplar = "Order the following adjectives by their sentiment. Adjectives: {mark}A{mark} great, {mark}B{mark} bad, {mark}C{mark} awful, {mark}D{mark} awesome. The correct ordering is: D, A, B, C.\n".format(mark = option_marker)
    else:
        exemplar = ""

    if prompt_type == "pair":
        x_y = {"x_pred": "texts", "y_pred": "y"}
        mask_tok = "[MASK]"
        n_tuple = 2
        pair_type = "perm"
        for ranking in data:
            assert "{answ}" in ranking["prompt"], "need prompt_type=pair"
            a_str, _, y = write_pairs(prompt=ranking["prompt"], ents=ranking["ents"],
                                               ranks=ranking["ranks"],
                                               answ=list(ranking["tok"].keys()), pair_type=pair_type,
                                               n_tuple=n_tuple)

            texts, ys = [], []
            for i in range(0, len(a_str), 1):
                prompt = a_str[i]
                answer_key = list(ranking["tok"].keys())[0]
                masked_prompt = mask_tok.join(prompt.rsplit(answer_key, 1))  ## replace last mention
                texts.append(masked_prompt)
                ys.append(y[i])
            ranking["y"] = ys
            ranking["texts"] = texts
            ## add entity ids, even possible if no ranks given as labels
            pairs_rank = list(helpers.get_pairings(np.arange(0, len(ranking["ents"]), 1), pair_type=pair_type, n_tuple=n_tuple))
            ranking["ent_ids"] = list(map(lambda pair: (-1, pair), pairs_rank))
            ranking["x_y"] = x_y


    elif prompt_type == "list":
        x_y = {"x_pred": "texts", "y_pred": "ranks"}
        for ranking in data:
            prompt = ranking['prompt']

            prompt_keys = [fname for _, fname, _, _ in Formatter().parse(prompt)][:-1]
            if len(prompt_keys) >= 1:
                ents_ranks = list(zip(ranking['ents'], ranking['ranks']))
                random.shuffle(ents_ranks)
                ents, ranks = zip(*ents_ranks)  ## shuffle entities for prompt
                enum_str = list(ranking['tok'])[0]
                ent_enum_key = [fname for _, fname, _, _ in Formatter().parse(enum_str)][-1:]  ## get entity enumeration
                ent_string, enum_ent_rank = "", {}
                for i, (ent, rank) in enumerate(zip(ents, ranks)):
                    if len(ent_enum_key) >= 1:
                        if ent_enum_key[0] == "number":
                            number = str(i+1)
                            enum_str_filled = enum_str.format(number=number)
                        elif ent_enum_key[0] == "alpha":
                            alpha = chr(i + 65)
                            if int(i) > 25:
                                alpha = str(int(i) - 25) ## continue counting with numbers 1, 2, 3...
                            enum_str_filled = enum_str.format(alpha=alpha)
                        else:
                            raise f"ent_enum_key in tok must be either num or alpha {ranking['tok']}"
                        enum_ent_rank[enum_str_filled] = rank
                    else:
                        enum_str_filled = ""
                        enum_ent_rank[ent] = rank
                    ent_string += f'{option_marker}{enum_str_filled}{option_marker} {ent}, '

                ent_string = ent_string[:-2]  ## remove last comma and space
                prompt = prompt.format(x=ent_string)
            else:  ## no {x} in prompt at all
                enum_ent_rank = dict(zip(ranking['ents'], ranking['ranks']))

            ranking["enums"] = enum_ent_rank
            ranking["texts"] = exemplar + prompt
            ranking["x_y"] = x_y

    elif prompt_type == "point" or prompt_type == "point_scale":
        x_y = {"x_pred": "texts", "y_pred": "ranks"}
        scale_cands = ["0","1","2","3","4","5","6","7","8","9","10"]
        ## could also do [0,1], [0,1,2,3,4,5,6,7,8,9,10] or [small, large]
        for ranking in data:
            ranking["texts"] = write_singles(prompt= ranking['prompt'], ents=ranking['ents'], answ=["[MASK]"])
            ranking["x_y"] = x_y
            ranking["toks"] = scale_cands
            ranking["ent_ids"] = list(range(0, len(ranking['ents']), 1))
    return data


def write_singles(prompt:str="", ents:list=[], answ="", **kwargs):
    """
    for itemSingle S prompt (pointwise)
    """
    x_str = []
    for ent in ents:
        ent_str = prompt.format(x = ent, answ=answ[0])
        x_str.append(ent_str)
    return x_str
    

def write_pairs(prompt:str="", ents:list=[], answ:list=[], ranks:list=None, pair_type="perm", n_tuple:int=2):
    """
    for itemPair P prompt
    pair_type: either "perm" or "comb"
    n_tuple: number of items per pair—either 2 or 3
    """

    def build_string_pair(prompt, a_b, a_b_rank=None, a1b1a2b2=[], answ=[]):
        a_str = prompt.format(a=a_b[a1b1a2b2[0][0]], answ=answ[0], b=a_b[a1b1a2b2[0][1]])
        b_str = prompt.format(a=a_b[a1b1a2b2[1][0]], answ=answ[1], b=a_b[a1b1a2b2[1][1]])

        if ranks is not None: ## tricky function, the y_pair_label always indicates which element in tuple is larger
            if (a_b_rank[a1b1a2b2[0][0]] > a_b_rank[a1b1a2b2[0][1]]):
                y_pair_label = 0
                y_pair = (y_pair_label, (int(a_b_rank[a1b1a2b2[0][0]]), int(a_b_rank[a1b1a2b2[0][1]])))
            else:
                y_pair_label = 1
                y_pair = (y_pair_label, (int(a_b_rank[a1b1a2b2[0][0]]), int(a_b_rank[a1b1a2b2[0][1]])))
        else:
            y_pair_label = -1  ## if not rank is given, always set right answer to -1
            y_pair = (y_pair_label, (int(a_b_rank[a1b1a2b2[0][0]]), int(a_b_rank[a1b1a2b2[0][1]])))
        return a_str, b_str, y_pair

    pairs = helpers.get_pairings(ents, pair_type=pair_type, n_tuple=n_tuple)

    if ranks is not None:
        pairs_rank = list(helpers.get_pairings(ranks, pair_type=pair_type, n_tuple=n_tuple))
    else:  ## no ground truth ranks provided, so just assign ids
        pairs_rank = list(helpers.get_pairings(np.arange(0, len(ents), 1), pair_type=pair_type, n_tuple=n_tuple))

    a, b, y = [], [], []
    for i, pair in enumerate(pairs):
        a_str, b_str, y_binary = build_string_pair(prompt=prompt, a_b = pair, a_b_rank=pairs_rank[i], a1b1a2b2=[[0,1],[0,1]], answ=answ)
        a.append(a_str), b.append(b_str), y.append(y_binary)
    return (a, b, y)



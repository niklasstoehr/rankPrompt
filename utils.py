#utils: helper functions such as converting pairwise to global rankings


def get_pairings(x, pair_type:str="perm", n_tuple:int=2, get_argmax=False):
    if pair_type == "comb":
        x = list(itertools.combinations(x, n_tuple))
    if pair_type == "perm":
        x = list(itertools.permutations(x, n_tuple))
    if get_argmax:
        x = list(map(lambda pair: np.argmax(pair), x))
    return x


def pairwise_to_ranks(idx_list, a_b_list, float_list=None, tie=False):
    tie=False
    ent_wins = defaultdict(int)  ## dict to store wins
    ent_scores = defaultdict(float)

    if float_list is None:
        float_list = [1] * len(idx_list)

    for idx, score, (true_idx, ab) in zip(idx_list, float_list, a_b_list):  ## true_idx not used
        ent_scores[int(ab[int(idx)])] += score
        ent_scores[int(ab[2 + ~int(idx)])] += 0

        ent_wins[int(ab[int(idx)])] += 1
        ent_wins[int(ab[2 + ~int(idx)])] += 0

    if tie:
        ent_wins = dict(sorted(ent_wins.items(), key=lambda item: item[0], reverse=True))  ## sort by keys
        rank_pred = list(stats.rankdata(np.array(list(ent_wins.values())), method='dense') - 1)  ## [1,1,0] --> [2, 2, 1] ranking with ties
    else:
        ent_ids = list(ent_wins.keys())
        ent_win_sum = list(ent_wins.values())
        ent_score_sum = list(ent_scores.values())
        if len(np.unique(ent_win_sum)) < len(ent_wins.keys()):
            if len(np.unique(ent_score_sum)) < len(ent_wins.keys()):
                print(f"ranking has ties: wins: {ent_win_sum}, log-scores: {ent_score_sum}")
        ent_wins_logscores = list(zip(ent_ids, ent_win_sum, ent_score_sum))
        ent_wins_logscores.sort(key=lambda i: (i[1], i[2]), reverse=True)  ## sort by wins, then log-scores to break ties
        rank_pred = list(map(lambda x: x[0], ent_wins_logscores))  ## get entity ids but sorted, wins and log scores
    return rank_pred


## EVALUATION______________________________________________________-


def inspect(probe, ds, print_res=True, **kwargs):
    for ranking in ds:
        y_pred, rank_pred = probe.predict(ranking, return_y=False, **kwargs)
        rank_pred = rank_pred.detach().numpy()
        ranked_text = sorted(list(zip(rank_pred, ranking["ents"])), key=lambda x: x[0], reverse=True)
        if print_res:
            print(f"{ranking['prompt']}\n {ranked_text}\n")


def metric_pairwise(y_true: torch.tensor, y_pred: torch.tensor, print_res=[]):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().numpy()

    ## finding the flipped direction
    y_true, y_pred = np.array(y_true).round(decimals=0), np.array(y_pred).round(decimals=0)
    y_pred_flip = 1 - y_pred

    if accuracy_score(y_true, y_pred) < accuracy_score(y_true, y_pred_flip):
        y_pred = y_pred_flip

    accuracy = accuracy_score(y_true, y_pred)
    results = {"accuracy": round(accuracy,4)}
    if "per_ranking" in print_res:
        print(f"pairwise: {results}")
    return results, y_pred


def metric_ranks(ranks_true: torch.tensor, ranks_pred: torch.tensor, print_res=[]):
    if isinstance(ranks_pred, torch.Tensor):
        ranks_pred = ranks_pred.detach().numpy()
    if isinstance(ranks_true, torch.Tensor):
        ranks_true = ranks_true.detach().numpy()

    ## finding the flipped direction
    ranks_true, ranks_pred = np.array(ranks_true), np.array(ranks_pred)
    ranks_pred_flip = np.flip(ranks_pred, axis=-1)

    if stats.kendalltau(ranks_true, ranks_pred).statistic < stats.kendalltau(ranks_true, ranks_pred_flip).statistic:
        ranks_pred = ranks_pred_flip

    kendall = stats.kendalltau(ranks_true.squeeze(), ranks_pred.squeeze()).statistic
    if math.isnan(kendall):
        kendall = 0.0
    results = {"kendall": round(kendall,4)}
    if "per_ranking" in print_res:
        print(f"ranks: {results}")
    return results, ranks_pred


def evaluate(probe, ds, return_type=["return_res"], print_res=["all_rankings"], **kwargs):
    res_ranks, res_pairs, ranks_preds, y_preds = [], [], [], []
    for ranking in ds:
        rank_pred, y_pred, rank_true, y_true = probe.predict(ranking, return_y=True, **kwargs)
        res_rank, rank_pred_flipped = metric_ranks(rank_true, rank_pred, print_res)
        res_y, y_pred_flipped = metric_pairwise(y_true, y_pred, print_res)
        res_ranks.append(list(res_rank.values()))
        res_pairs.append(list(res_y.values()))
        ranks_preds.append(rank_pred_flipped)
        y_preds.append(y_pred_flipped)

    res_ranks = dict(zip(list(res_rank.keys()), np.stack(res_ranks).mean(-2).round(4)))
    res_pairs = dict(zip(list(res_y.keys()), np.stack(res_pairs).mean(-2).round(4)))

    if "all_rankings" in print_res:
        print(f"ranks: {res_ranks} \npairwise: {res_pairs}\n")
    if "return_pred" in return_type:
        return ranks_preds, y_preds
    if "return_res" in return_type:
        return {**res_ranks, **res_pairs}
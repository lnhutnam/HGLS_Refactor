import torch


def sort_and_rank(score, target):
    """Sorting and ranking

    Args:
        score (torch.Tensor): _description_
        target (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def sort_and_rank_time_filter(batch_a, batch_r, score, target, total_triplets):
    """Sorting and ranking filer by groud truth in the same time snapshot not all ground truth

    Args:
        batch_a (torch.Tensor): _description_
        batch_r (torch.Tensor): _description_
        score (torch.Tensor): _description_
        target (torch.Tensor): _description_
        total_triplets (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    for i in range(len(batch_a)):
        ground = indices[i]
    indices = indices[:, 1].view(-1)
    return indices


def sort_and_rank_filter(batch_a, batch_r, score, target, all_ans):
    """Sorting and ranking with filter

    Args:
        batch_a (torch.Tensor): _description_
        batch_r (torch.Tensor): _description_
        score (torch.Tensor): _description_
        target (torch.Tensor): _description_
        all_ans (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    for i in range(len(batch_a)):
        ans = target[i]
        b_multi = list(all_ans[batch_a[i].item()][batch_r[i].item()])
        ground = score[i][ans]
        score[i][b_multi] = 0
        score[i][ans] = ground
    _, indices = torch.sort(
        score, dim=1, descending=True
    )  # indices : [B, number entity]
    indices = torch.nonzero(
        indices == target.view(-1, 1)
    )  # indices : [B, 2] The first column is incrementing, and the second column indicates the position of the corresponding answer entity id in each row.
    indices = indices[:, 1].view(-1)
    return indices


def filter_score(test_triples, score, all_ans):
    """_summary_

    Args:
        test_triples (_type_): _description_
        score (_type_): _description_
        all_ans (_type_): _description_

    Returns:
        _type_: _description_
    """
    if all_ans is None:
        return score

    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple[[0, 1, 2]]
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score


def filter_score_r(test_triples, score, all_ans):
    """_summary_

    Args:
        test_triples (_type_): _description_
        score (_type_): _description_
        all_ans (_type_): _description_

    Returns:
        _type_: _description_
    """
    if all_ans is None:
        return score

    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][t.item()])
        # print(h, r, t)
        # print(ans)
        ans.remove(r.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score


def get_total_rank(test_triples, score, all_ans, eval_bz, rel_predict=0):
    """_summary_

    Args:
        test_triples (_type_): _description_
        score (_type_): _description_
        all_ans (_type_): _description_
        eval_bz (_type_): _description_
        rel_predict (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_rank = []

    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        if rel_predict == 1:
            target = test_triples[batch_start:batch_end, 1]
        elif rel_predict == 2:
            target = test_triples[batch_start:batch_end, 0]
        else:
            target = test_triples[batch_start:batch_end, 2]
        rank.append(sort_and_rank(score_batch, target))

        if rel_predict:
            filter_score_batch = filter_score_r(triples_batch, score_batch, all_ans)
        else:
            filter_score_batch = filter_score(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1  # change to 1-indexed
    filter_rank += 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank


def stat_ranks(rank_list):
    """Stat information for ranking

    Args:
        rank_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)
    hit_metric = []

    mrr = torch.mean(1.0 / total_rank.float())
    # print("MRR ({}): {:.6f}".format(method, mrr.item()))
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        # if not test:
        #     print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
        hit_metric.append(avg_count.item())
    return mrr, hit_metric[0], hit_metric[1], hit_metric[2]

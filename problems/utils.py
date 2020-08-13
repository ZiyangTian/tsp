import random
import torch


def opt2_randomly(*gene_data):
    size = gene_data[0].size().numpy()
    index = torch.tensor(random.sample(range(size[1]), 2) for _ in range(size[0]))

    def exchange(tensor):
        exchanged_index = torch.stack([index[:, 1], index[:, 0]], dim=1)
        full_index = torch.stack([torch.arange(size[1]) for _ in torch.arange(size[0])])
        exchanged_full_index = full_index.scatter(1, index, exchanged_index)
        exchanged_full_index, _ = torch.broadcast_tensors(exchanged_full_index, torch.empty(*size))
        exchanged = tensor.gather(1, exchanged_full_index)
        return exchanged

    ans = tuple(map(exchange, gene_data))
    if len(ans) == 1:
        return ans[0]
    return ans

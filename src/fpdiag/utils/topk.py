import heapq


class StreamingTopK:
    def __init__(self, k: int):
        self.k = int(k)
        self._heap = []

    def update(self, values, offset: int = 0):
        import torch
        flat = torch.as_tensor(values).detach().abs().reshape(-1).cpu()
        for local, value in enumerate(flat.tolist()):
            item = (float(value), int(offset + local))
            if len(self._heap) < self.k:
                heapq.heappush(self._heap, item)
            elif item > self._heap[0]:
                heapq.heapreplace(self._heap, item)

    def result(self):
        import torch
        ordered = sorted(self._heap, reverse=True)
        return torch.tensor([x[0] for x in ordered]), torch.tensor([x[1] for x in ordered])

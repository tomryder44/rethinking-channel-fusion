import numpy as np
from torch.utils.data import Sampler

import logging
logger = logging.getLogger(__name__)


class DomainSampler(Sampler):

    def __init__(self, dataset, samples_per_domain):
        self.dataset = dataset
        self.samples_per_domain = samples_per_domain
        domain_labels = dataset.domain_labels.cpu().numpy()
        self.domains = np.unique(domain_labels)
        self.domain_indices = {d: np.where(domain_labels == d)[0] for d in self.domains}
        for d, idxs in self.domain_indices.items():
            logger.debug(f"Domain {d}: {len(idxs)} samples")
        total = sum(len(idxs) for idxs in self.domain_indices.values())
        batch = samples_per_domain * len(self.domains)
        self.num_batches = total // batch
        logger.debug(f"num batches: {self.num_batches}")

    def __iter__(self):
        pos = {d: 0 for d in self.domains}
        shuffled = {d: np.random.permutation(idxs) for d, idxs in self.domain_indices.items()}
        for _ in range(self.num_batches):
            batch = []
            for d in self.domains:
                need = self.samples_per_domain
                if pos[d] + need > len(shuffled[d]):
                    shuffled[d] = np.random.permutation(self.domain_indices[d])
                    pos[d] = 0
                batch.extend(shuffled[d][pos[d]:pos[d] + need])
                pos[d] += need
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


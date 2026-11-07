"""Named predecessor-inspired raw operators and matching descriptor controls."""
import numpy as np
import scipy.sparse as sp
from bundlefuse.metrics import ranking_metrics


def standardize(values):
    values = np.asarray(values, np.float32)
    return (values-values.mean(0, keepdims=True))/(values.std(0, ddof=1, keepdims=True)+1e-9)


class AmplifiedNeighbors:
    """All or top-k historical cosine neighbors, with seed-frequency weights.

    A controlled predecessor-inspired implementation, NOT an exact Kelen reproduction:
    no positions or min/max normalization. Every variant is explicitly recorded.
    No dense item x item or item x item x item tensor is constructed.
    """
    def __init__(self, graph):
        self.B = graph.B_bin
        self.a = graph.inv_sqrt_f
        self.volume = np.sqrt(graph.bsize)

    def overlaps(self, seeds):
        x=sp.csr_matrix((self.a[seeds], (seeds, np.zeros(len(seeds),int))),
                        shape=(self.B.shape[1],1))
        return np.asarray((self.B@x).todense()).ravel()/(self.volume*np.sqrt(len(seeds)))

    def score(self, seeds, power, neighbors):
        overlap=self.overlaps(seeds)
        if neighbors is not None and neighbors < len(overlap):
            # Stable selection includes deterministic ties at the boundary.
            ids=np.arange(len(overlap));keep=np.lexsort((ids,-overlap))[:neighbors]
            mask=np.zeros(len(overlap),bool);mask[keep]=True;overlap[~mask]=0
        return np.asarray(self.B.T @ np.power(overlap,power)).ravel().astype(np.float32)

    def select(self, pairs, indices, powers, neighbors, log):
        grid=[dict(power=p,neighbors=k) for p in powers for k in neighbors]
        totals=np.zeros(len(grid),np.float64)
        for number,ix in enumerate(indices):
            seeds,gt=pairs[ix]
            overlap=self.overlaps(seeds)
            order=np.lexsort((np.arange(len(overlap)),-overlap))
            for j,cfg in enumerate(grid):
                weights=np.power(overlap,cfg['power'])
                if cfg['neighbors'] is not None: weights[order[cfg['neighbors']:]]=0
                score=np.asarray(self.B.T@weights).ravel().astype(np.float32)
                totals[j]+=ranking_metrics(score[:,None],[(seeds,gt)],Ks=(20,))[20][1]
            if (number+1)%100==0:log(f'amplified-neighbor fitting selection {number+1}/{len(indices)}')
        means=totals/len(indices)
        best=int(np.argmax(means))
        return dict(grid[best],fit_ndcg20=float(means[best]),
                    grid=[dict(c,ndcg20=float(v)) for c,v in zip(grid,means)],
                    definition='B.T @ truncated[(B @ D x)/(sqrt(bundle_size)*sqrt(seed_count))]^power')

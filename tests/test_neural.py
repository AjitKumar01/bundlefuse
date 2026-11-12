"""Behavioral checks for the independent neural experiment."""
import tempfile
import unittest
import shutil
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from experiments.neural.data import make_batch, completion_loss, pad_seeds
from experiments.neural.model import BundleNetwork, ModelConfig
from experiments.neural.evaluation import evaluate
from experiments.neural.training import save_checkpoint, fit


class IndependentNeuralTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.features = np.random.default_rng(1).normal(size=(12, 7)).astype('float32')
        self.cats = np.arange(12) % 3
        self.freq = np.r_[np.ones(10)*3, 0, 0].astype('float32')

    def model(self, variant='contextual'):
        return BundleNetwork(self.features, self.cats, self.freq,
                             ModelConfig(variant=variant, dim=8, layers=1, heads=2,
                                         dropout=0., identity_dropout=0.))

    def test_set_permutation_and_padding_invariance(self):
        for variant in ('dot','relations','contextual'):
            model = self.model(variant).eval()
            candidates = torch.tensor([[4,5,6]])
            a, mask = pad_seeds([[0,1,2]], 12)
            baseline = model(a, mask, candidates)
            b, valid = pad_seeds([[2,0,1], [0,1,2,3,4]], 12)
            padded = model(b, valid, candidates.expand(2,-1))[:1]
            torch.testing.assert_close(baseline, padded, atol=2e-6, rtol=2e-6)

    def test_candidate_chunk_and_checkpoint_parity(self):
        model = self.model().eval()
        pairs = [([0,1], [2,3]), ([2], [5])]
        a = evaluate(model, pairs, batch_size=1, candidate_chunk=3)
        b = evaluate(model, pairs, batch_size=2, candidate_chunk=12)
        self.assertEqual(a['queries'], b['queries'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            save_checkpoint(path, {'model':model.state_dict()})
            restored = self.model()
            restored.load_state_dict(torch.load(path, weights_only=True)['model'])
            c = evaluate(restored, pairs, candidate_chunk=4)
            self.assertEqual(a['queries'], c['queries'])

    def test_sampling_excludes_entire_bundle_and_retains_targets(self):
        bundles = [[0,1,2,3], [4,5]]
        rng = np.random.default_rng(12)
        for _ in range(20):
            seeds, valid, candidates, positives, negatives = make_batch(bundles,12,rng,negative_pool=20)
            for i, bundle in enumerate(bundles):
                observed = set(seeds[i,valid[i]].tolist())
                hidden = set(candidates[i,positives[i]].tolist())
                negative = set(candidates[i,negatives[i]].tolist())
                self.assertEqual(observed | hidden, set(bundle))
                self.assertFalse(observed & hidden)
                self.assertFalse(negative & set(bundle))
                self.assertTrue(observed and hidden and negative)

    def test_loss_ignores_padding_and_does_not_compete_positives(self):
        scores = torch.tensor([[1.,2.,-1.,.5,1000.]], requires_grad=True)
        positive = torch.tensor([[True,True,False,False,False]])
        negative = torch.tensor([[False,False,True,True,False]])
        loss = completion_loss(scores, positive, negative, hard=False, negatives=2)
        expected = torch.stack([torch.logsumexp(scores[0,[i,2,3]],0)-scores[0,i] for i in [0,1]]).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(scores.grad[0,4], 0.)
        self.assertTrue(bool((scores.grad[0,:2] < 0).all()))

    def test_no_random_untrained_identity_for_unseen_items(self):
        model = self.model().eval()
        before = model.encode_items(torch.tensor([10,11])).detach()
        with torch.no_grad():
            model.identity.weight[10:12] += 100
        torch.testing.assert_close(before, model.encode_items(torch.tensor([10,11])))

    def test_can_learn_synthetic_completion(self):
        model = self.model('contextual').train()
        seeds, valid = pad_seeds([[0],[2],[4],[6]],12)
        candidates = torch.tensor([[1,3,5,7],[3,1,5,7],[5,1,3,7],[7,1,3,5]])
        positive = torch.zeros_like(candidates,dtype=torch.bool); positive[:,0] = True
        negative = ~positive
        optimizer = torch.optim.Adam(model.parameters(),lr=.01)
        losses = []
        for _ in range(80):
            optimizer.zero_grad()
            loss = completion_loss(model(seeds,valid,candidates),positive,negative,hard=False,negatives=3)
            losses.append(float(loss.detach())); loss.backward(); optimizer.step()
        self.assertLess(losses[-1], .2*losses[0])
        self.assertTrue(bool((model(seeds,valid,candidates).argmax(1) == 0).all()))

    def test_resume_restores_optimizer_and_rng_exactly(self):
        ds = SimpleNamespace(n_items=12, train=[[0,1,2],[2,3],[4,5],[6,7,8]],
                             val_pairs=[([0],[1]),([4],[5])])
        config = dict(dim=8, epochs=4, lr=.001, weight_decay=.0001, batch_size=2,
                      eval_batch=2, candidate_chunk=4, negative_pool=8, negatives=4,
                      eval_every=2, patience=6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fit(ds,self.features,self.cats,self.freq,'contextual',0,config,
                root/'full',root/'full_output',lambda _:None)
            (root/'resume').mkdir()
            shutil.copy2(root/'full/epoch_0002.pt',root/'resume/epoch_0002.pt')
            fit(ds,self.features,self.cats,self.freq,'contextual',0,config,
                root/'resume',root/'resume_output',lambda _:None)
            a = torch.load(root/'full/epoch_0004.pt',weights_only=False)
            b = torch.load(root/'resume/epoch_0004.pt',weights_only=False)
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key],b['model'][key]),key)
            self.assertEqual(a['best_epoch'],b['best_epoch'])
            self.assertEqual(a['history'][-1]['loss'],b['history'][-1]['loss'])


if __name__ == '__main__':
    unittest.main()

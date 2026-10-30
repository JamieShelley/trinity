"""Focused deterministic-domain and evidence-preservation regressions."""
from __future__ import annotations
import unittest
import torch
from tools.nsamdr.neural.benchmark_nsamdr_v178_contrast import registered_batch
from tools.nsamdr.neural.diagnose_nsamdr_v178_normal_domain import perturb
from tools.nsamdr.neural.train_nsamdr_v178_replay import replay_batch
from tools.nsamdr.neural.train_nsamdr_v178_normal_variation import variation_batch
from tools.nsamdr.neural.v178.contour_model import physical_edge_evidence

class DomainDiagnostics(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(min(4,torch.get_num_threads()))
  self.a,self.n,self.m,self.t=registered_batch(7017832,torch.device('cpu'))
 def test_nuisance_is_seeded(self):
  x=perturb(self.a,self.n,self.m,123,kind='iid')
  y=perturb(self.a,self.n,self.m,123,kind='iid')
  self.assertTrue(all(torch.equal(a,b) for a,b in zip(x,y)))
 def test_nuisance_changes_with_seed(self):
  x,_=perturb(self.a,self.n,self.m,123,kind='smooth')
  y,_=perturb(self.a,self.n,self.m,124,kind='smooth')
  self.assertFalse(torch.equal(x,y))
 def test_nuisance_does_not_mutate_input(self):
  a,n,m,t=[x.clone() for x in (self.a,self.n,self.m,self.t)]
  perturbed,new_m=perturb(self.a,self.n,self.m,123)
  for x,y in zip((self.a,self.n,self.m,self.t),(a,n,m,t)):
   self.assertTrue(torch.equal(x,y))
  self.assertFalse(torch.equal(perturbed,n))
  self.assertTrue(torch.equal(new_m[:,1:],m[:,1:]))
 def test_edge_is_recomputed_from_native_observations(self):
  for kind in ('iid','smooth'):
   n,m=perturb(self.a,self.n,self.m,124,kind=kind)
   self.assertTrue(torch.equal(m[:,:1],physical_edge_evidence(self.a,n)))
 def test_nuisance_valid_range_and_finite(self):
  for kind in ('iid','smooth'):
   n,m=perturb(self.a,self.n,self.m,124,kind=kind)
   self.assertTrue(torch.isfinite(n).all())
   self.assertTrue(torch.isfinite(m).all())
   self.assertLessEqual(float(n.abs().max()),1.0)
 def test_nuisance_independent_of_global_rng(self):
  torch.manual_seed(53)
  first=perturb(self.a,self.n,self.m,124,kind='smooth')
  torch.manual_seed(995)
  second=perturb(self.a,self.n,self.m,124,kind='smooth')
  self.assertTrue(all(torch.equal(a,b) for a,b in zip(first,second)))
 def test_replay_determinism(self):
  left=replay_batch(16,17862,torch.device('cpu'))
  right=replay_batch(16,17862,torch.device('cpu'))
  self.assertTrue(all(torch.equal(a,b) for a,b in zip(left,right)))
 def test_variation_batch_determinism(self):
  left=variation_batch(17872,0,torch.device('cpu'))
  right=variation_batch(17872,0,torch.device('cpu'))
  self.assertTrue(all(torch.equal(a,b) for a,b in zip(left,right)))
 def test_variation_changes_only_normal_and_edge_for_half_batch(self):
  for step in (0,1):
   seed=17872+8191*step
   original=replay_batch(32,seed,torch.device('cpu'),composed_slot=step%4)
   variation=variation_batch(seed,step,torch.device('cpu'))
   self.assertTrue(torch.equal(original[0],variation[0]))
   self.assertTrue(torch.equal(original[3],variation[3]))
   changed=0
   for i in range(4):
    if not torch.equal(original[1][i],variation[1][i]):
     changed+=1
     self.assertTrue(torch.equal(variation[2][i,1:],original[2][i,1:]))
     edge=physical_edge_evidence(variation[0][i:i+1],variation[1][i:i+1])
     self.assertTrue(torch.equal(variation[2][i:i+1,:1],edge))
    else:
     self.assertTrue(torch.equal(original[2][i],variation[2][i]))
   self.assertEqual(changed,2)

if __name__=='__main__':unittest.main()

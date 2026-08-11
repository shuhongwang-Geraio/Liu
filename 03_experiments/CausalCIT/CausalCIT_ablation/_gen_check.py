import argparse
import os
import run_large as rl

ns = argparse.Namespace(datasets=['traffic'], variants=['full_v2_fixed'], seeds='42',
                        dataset_dir=None, num_shards=1, entropy_weight=0.0,
                        n_envs=2, rff_dim=64, prior_weight=0.1, temperature=1.0,
                        output_dir='./_tmp_gencheck')
shards = rl.gen_jobs(ns)
jobs = [j for sh in shards for j in sh]  # 展平
j = jobs[0]
mk = j['model_kwargs']
print('RESULT variant=%s n_envs=%s rff_dim=%s prior_weight=%s temperature=%s njobs=%d'
      % (j['variant'], mk.get('n_envs'), mk.get('rff_dim'),
         mk.get('prior_weight'), mk.get('temperature'), len(jobs)))
d = rl.resolve_dataset_dir(None)
print('DATADIR=%s EXISTS=%s' % (d, os.path.isdir(d) and bool(os.listdir(d))))

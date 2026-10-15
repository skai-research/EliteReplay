"""Two-epoch, response-averaged archive SFT. No generation, rewards, or filtering."""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def encode(tokenizer,row):
    # Match CHORD's existing auxiliary SFT prompt/artifact/EOS contract exactly.
    from elitereplay.archive.trinity_support import render_sft_with_terminal_eos
    ids,start,_=render_sft_with_terminal_eos(tokenizer,row['messages'],enable_thinking=False)
    return {'ids':ids,'start':start,'response_tokens':len(ids)-start}


def chunk_loss(head,hidden,targets,chunk=128,checkpoint=True):
    """Exactly mean assistant-token CE, with bounded vocabulary logits memory."""
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint as remat
    total=hidden.new_zeros((),dtype=torch.float32)
    def ce(h,t):return F.cross_entropy(head(h).float(),t,reduction='sum')
    for begin in range(0,len(targets),chunk):
        h=hidden[begin:begin+chunk];t=targets[begin:begin+chunk]
        total=total+(remat(ce,h,t,use_reentrant=False) if checkpoint else ce(h,t))
    return total/len(targets)


def prepare(config):
    from transformers import AutoTokenizer
    c=json.loads(Path(config).read_text());out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    if sha(c['archive'])!=c['archive_sha256']:raise ValueError('Frozen archive changed')
    tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True,trust_remote_code=True)
    rows=[json.loads(x) for x in Path(c['archive']).read_text().splitlines()]
    encoded=[encode(tok,row) for row in rows]
    if max(len(x['ids']) for x in encoded)>c['max_sequence_tokens']:raise ValueError('Sequence ceiling exceeded; no truncation allowed')
    for i,x in enumerate(encoded):x['row_index']=i
    tokens=out/'tokens.json';atomic(tokens,encoded)
    audit={'status':'passed','config_sha256':sha(config),'archive_sha256':c['archive_sha256'],'tokens_sha256':sha(tokens),'rows':len(rows),'distinct_candidates':len({r['metadata']['candidate_id'] for r in rows}),'questions':len({r['metadata']['problem_id'] for r in rows}),'max_sequence_tokens':max(len(x['ids']) for x in encoded),'max_response_tokens':max(x['response_tokens'] for x in encoded),'duplicate_rows_preserved':True,'epochs':2,'updates_per_epoch':math.ceil(len(rows)/c['batch_size']),'base_files':{str(p):sha(p) for p in Path(c['model']).glob('*.safetensors')}}
    atomic(out/'prepared.json',audit);print(json.dumps(audit),flush=True)


def train(config):
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    c=json.loads(Path(config).read_text());out=Path(c['output']);audit=json.loads((out/'prepared.json').read_text())
    if audit['config_sha256']!=sha(config) or audit['tokens_sha256']!=sha(out/'tokens.json') or sha(c['archive'])!=c['archive_sha256']:
        raise ValueError('Training inputs changed')
    for file,digest in audit['base_files'].items():
        if sha(file)!=digest:raise ValueError('Pinned base changed')
    rows=json.loads((out/'tokens.json').read_text())
    lock=(out/'writer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'complete.json').exists():print('Already completed');return
    torch.set_num_threads(4);random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    latest=max((e for e in (1,2) if (out/f'epoch-{e}'/'checkpoint_complete.json').exists()),default=0)
    source=out/f'epoch-{latest}' if latest else Path(c['model'])
    torch.cuda.set_device(0)
    capacity=torch.cuda.get_device_properties(0).total_memory/2**30
    if capacity<c.get('minimum_gpu_gib',75):
        raise RuntimeError(f'SFT requires an 80GB GPU; allocated capacity is {capacity:.2f} GiB')
    print(json.dumps({'gpu':torch.cuda.get_device_name(0),'capacity_gib':capacity}),flush=True)
    from elitereplay.training.kernels import require_fast_path
    require_fast_path()
    # FP32 parameters/moments, BF16 computation; all language parameters trainable.
    model=Qwen3_5ForConditionalGeneration.from_pretrained(source,local_files_only=True,trust_remote_code=True,dtype=torch.float32,attn_implementation='sdpa').to('cuda')
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False)
    per_epoch=math.ceil(len(rows)/c['batch_size']);total=2*per_epoch;warmup=max(1,math.ceil(total*.05))
    def factor(step):
        if step<warmup:return (step+1)/warmup
        progress=(step-warmup)/max(1,total-warmup)
        return .1+.9*.5*(1+math.cos(math.pi*min(1,progress)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    global_step=0
    if latest:
        state=torch.load(source/'training_state.pt',map_location='cpu',weights_only=False)
        if state['config_sha256']!=sha(config):raise ValueError('Checkpoint config mismatch')
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);global_step=state['global_step']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);random.setstate(state['python_rng']);del state
    def loss(row):
        ids=torch.tensor([row['ids']],device='cuda')
        # A single sequence per microbatch prevents cross-example attention.
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            hidden=model.model(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state[0]
            start=row['start'];return chunk_loss(model.lm_head,hidden[start-1:-1],ids[0,start:])
    if not latest:
        # Exercise accumulation both before and after Adam moments are resident.
        # A zero-LR probe leaves weights unchanged; discard probe moments and RNG.
        test=max(rows,key=lambda r:len(r['ids']));torch.cuda.reset_peak_memory_stats()
        rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state();python_rng=random.getstate()
        saved_lrs=[group['lr'] for group in optimizer.param_groups]
        for group in optimizer.param_groups:group['lr']=0.
        for phase in range(2):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(2):
                value=loss(test);(value/2).backward()
                if not math.isfinite(value.item()):raise ValueError('Invalid SFT loss')
                loss_value=value.item();del value
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if not 0<float(norm):raise ValueError('Invalid SFT gradient')
            if phase==0:optimizer.step()
        atomic(out/'gpu_check.json',{'status':'passed','loss':loss_value,'grad_norm':float(norm),'sequence_tokens':len(test['ids']),'peak_gib':torch.cuda.max_memory_allocated()/2**30,'capacity_gib':capacity,'accumulation_microbatches':2,'adam_state_resident_backward':True,'probe_lr':0})
        optimizer.zero_grad(set_to_none=True);optimizer.state.clear()
        for group,lr in zip(optimizer.param_groups,saved_lrs):group['lr']=lr
        torch.set_rng_state(rng);torch.cuda.set_rng_state(cuda_rng);random.setstate(python_rng)
        torch.cuda.empty_cache()
    tokenizer=AutoTokenizer.from_pretrained(c['model'],local_files_only=True,trust_remote_code=True)
    for epoch in range(latest+1,3):
        order=list(range(len(rows)));random.Random(c['seed']+epoch).shuffle(order)
        atomic(out/f'epoch-{epoch}-order.json',order)
        for start in range(0,len(order),c['batch_size']):
            indices=order[start:start+c['batch_size']];t=time.monotonic();value=0.;optimizer.zero_grad(set_to_none=True)
            for idx in indices:
                ce=loss(rows[idx]);(ce/len(indices)).backward();value+=float(ce.detach())/len(indices)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if not math.isfinite(value):raise ValueError('Nonfinite loss')
            lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step();global_step+=1
            record={'epoch':epoch,'step':global_step,'rows_in_epoch':start+len(indices),'loss':value,'grad_norm':float(norm),'lr':lr,'seconds':time.monotonic()-t,'peak_gib':torch.cuda.max_memory_allocated()/2**30}
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        optimizer.zero_grad(set_to_none=True)
        staging=out/f'epoch-{epoch}.building-{os.getpid()}';staging.mkdir()
        model.save_pretrained(staging,safe_serialization=True,max_shard_size='4GB');tokenizer.save_pretrained(staging)
        torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'global_step':global_step,'epoch':epoch,'config_sha256':sha(config),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'python_rng':random.getstate(),'consumed_rows_per_epoch':len(rows)},staging/'training_state.pt')
        files={p.name:sha(p) for p in staging.iterdir() if p.is_file()}
        atomic(staging/'checkpoint_complete.json',{'epoch':epoch,'step':global_step,'config_sha256':sha(config),'archive_sha256':c['archive_sha256'],'files_sha256':files})
        staging.rename(out/f'epoch-{epoch}')
        print(json.dumps({'saved_epoch':epoch,'path':str(out/f'epoch-{epoch}')}),flush=True)
    atomic(out/'complete.json',{'epochs':2,'updates':global_step,'rows_per_epoch':len(rows),'total_presentations':2*len(rows),'checkpoint':str(out/'epoch-2')})


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train']);p.add_argument('config');a=p.parse_args()
    (prepare if a.action=='prepare' else train)(a.config)

if __name__=='__main__':main()

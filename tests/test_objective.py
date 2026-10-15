import pytest
torch = pytest.importorskip("torch")
from types import SimpleNamespace
from evorl.baselines import grpo, maxrl, objective
from evorl.baselines.batching import arrange
from evorl.baselines.config import METHODS

def test_maxrl_weights_hard_groups_more_than_grpo():
    r = [1.] + [0.] * 7
    assert maxrl.advantages(r)[0] > grpo.advantages(r)[0]
    for implementation in (maxrl.advantages, grpo.advantages):
        with pytest.raises(ValueError, match="binary"):
            implementation([.5, 1])


@pytest.mark.parametrize("method", METHODS)
def test_microbatch_and_dp_gradient_invariance(method):
    mixed = method in ("IID_Replay", "EliteReplay", "dph_f", "dph_js")
    expert = torch.tensor([False, False, mixed, False, False, mixed])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.float64)
    old = torch.full((6, 3), -2., dtype=torch.float64)
    logp = (old + torch.linspace(-.4, .4, 18).reshape(6, 3)).requires_grad_()
    advantage = torch.tensor([[1.], [-1.], [.5], [-1.], [1.], [-.5]], dtype=torch.float64).expand(6, 3)
    online_count = (mask * (~expert[:, None])).sum().expand(6)
    aux_count = (mask * expert[:, None]).sum().clamp_min(1).expand(6)
    def calculate(indices, dp):
        return objective.loss(logp[indices], old[indices], mask[indices], advantage[indices], expert[indices],
            online_count[indices], aux_count[indices], method=method, dp_size=dp)[0]
    full = calculate(slice(None), 1)
    # Unequal microbatches on two ranks; emulate summed backward then DP mean.
    micro = sum(calculate(i, 2) for i in (slice(0, 1), slice(1, 3), slice(3, 5), slice(5, 6))) / 2
    torch.testing.assert_close(full, micro)
    torch.testing.assert_close(torch.autograd.grad(full, logp)[0], torch.autograd.grad(micro, logp)[0])


def test_layout_gives_each_update_its_global_source_normalizers():
    def experience(source, group, length):
        return SimpleNamespace(info={"is_expert": source, "group": group}, action_mask=torch.ones(length))
    groups = [[experience(False, i, (i % 3) + 1) for _ in range(8)] for i in range(8)]
    aux = [experience(True, "aux", i + 1) for i in range(8)]
    rows = arrange(groups, aux, prompts_per_update=4, dp_size=2, seed=42, iteration=1)
    ranks = [rows[:36], rows[36:]]
    for update in (0, 1):
        batch = [e for rank in ranks for e in rank[update * 18:(update + 1) * 18]]
        assert len(batch) == 36
        assert sum(e.info["is_expert"] for e in batch) == 4
        for source, key in ((False, "online_token_count"), (True, "auxiliary_token_count")):
            expected = sum(int(e.action_mask.sum()) for e in batch if e.info["is_expert"] == source)
            assert {e.info[key] for e in batch} == {expected}
        assert len({e.info["group"] for e in batch if not e.info["is_expert"]}) == 4



def test_response_mean_replay_weights_and_microbatch_gradients():
    # Unequal response lengths distinguish response mean from token mean.
    logp = torch.tensor([[-1., -2., -3.], [-4., -6., 0.], [-2., 0., 0.]], requires_grad=True)
    mask = torch.tensor([[1.,1.,1.], [1.,1.,0.], [1.,0.,0.]])
    expert = torch.tensor([False, True, True])
    old = logp.detach().clone()
    advantage = torch.ones_like(logp)
    online_count = torch.full((3,),3.)
    aux_count = torch.full((3,),3.)
    response_count = torch.full((3,),2.)
    def calc(indices, dp):
        return objective.loss(logp[indices],old[indices],mask[indices],advantage[indices],expert[indices],
            online_count[indices],aux_count[indices],method='EliteReplay',dp_size=dp,
            sft_weight=.05,sft_reduction='response_mean',auxiliary_response_count=response_count[indices])[0]
    full = calc(slice(None),1)
    # RL=-1; response-mean NLL=(5+2)/2=3.5.
    torch.testing.assert_close(full,torch.tensor(-.95+.05*3.5))
    micro = sum(calc(slice(i,i+1),2) for i in range(3))/2
    torch.testing.assert_close(torch.autograd.grad(full,logp)[0],torch.autograd.grad(micro,logp)[0])

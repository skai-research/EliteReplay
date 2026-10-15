import copy, hashlib, json, math, sqlite3
from pathlib import Path
from types import SimpleNamespace
import pytest
from evorl.evaluation.config import ROOT, atomic_json, digest, load, selected, sha256
from evorl.evaluation.data import question_key
from evorl.evaluation.engine import files_for, pass_at_k, run, sample_seed, score_sample, summarize
from evorl.evaluation.models import model_files

@pytest.fixture
def config(tmp_path):
    c = selected(load(ROOT / "configs/evaluation/bird.yaml"), "base")
    c["sampling"].update(samples_per_question=64, k_values=[1,2,4,8,16,32,64])
    c["data"].update(expected_questions=2, prepared_dir=str(tmp_path / "prepared"), training_data_root=str(tmp_path / "training"))
    c["output_root"] = str(tmp_path / "output")
    c["runtime"].update(launcher="native", verifier_workers=2)
    return c


@pytest.fixture
def items():
    return [{"problem_id": f"bird:official_dev:{i}", "question_key": question_key(f"Question {i}"),
             "prompt": [{"role": "user", "content": f"Question {i}"}],
             "reward_model": {"ground_truth": json.dumps({"inputs": ["1"], "outputs": ["1"]})},
             "extra_info": {"difficulty": "EASY"}} for i in range(2)]


class FakeVerifier:
    def parse_strict(self, text):
        return SimpleNamespace(accepted=True, canonical_response=text)

    def verify(self, item, response):
        return SimpleNamespace(fully_correct=response == "correct", executable=True, error=None, metadata={})


class FakeGenerator:
    calls = []

    def __init__(self, *args):
        pass

    def generate(self, requests):
        self.calls.extend((r["problem_id"], r["sample_id"]) for r in requests)
        return [{"text": "correct" if r["sample_id"] % 2 == 0 else "wrong", "finish_reason": "stop",
                 "response_tokens": 2, "prompt_tokens": 3} for r in requests]


@pytest.mark.parametrize("n,c,k", [(64,0,1),(64,0,64),(64,1,1),(64,1,64),(64,32,4),(64,64,32),(8,3,2)])
def test_passk_exact_combinatorial_estimator(n,c,k):
    expected = 1 - (math.comb(n-c,k) / math.comb(n,k) if n-c >= k else 0)
    assert pass_at_k(n,c,k) == pytest.approx(expected)


@pytest.mark.parametrize("args", [(4,1,8),(4,5,1),(0,0,1),(4,-1,1),(4,1,0)])
def test_passk_refuses_invalid_counts(args):
    with pytest.raises(ValueError):pass_at_k(*args)


def test_full_run_and_resume_never_regenerate(config,items,tmp_path):
    FakeGenerator.calls=[]
    report=run(config,"base",items,tmp_path,generator_factory=FakeGenerator,verifier=FakeVerifier())
    assert len(FakeGenerator.calls)==128 and len(set(FakeGenerator.calls))==128
    assert report["questions"]==2 and report["pass_at_k"]["1"]==0.5 and report["pass_at_k"]["64"]==1
    assert run(config,"base",items,tmp_path,generator_factory=FakeGenerator,verifier=FakeVerifier())==report
    assert len(FakeGenerator.calls)==128
    assert json.loads((tmp_path/"summary.json").read_text())["status"]=="complete"


def test_shards_cover_global_questions_without_collisions(config,items,tmp_path):
    config["sampling"]["questions_per_batch"] = 1
    FakeGenerator.calls = []
    first = run(config, "base", items, tmp_path, num_shards=2, shard_index=0,
                generator_factory=FakeGenerator, verifier=FakeVerifier())
    second = run(config, "base", items, tmp_path, num_shards=2, shard_index=1,
                 generator_factory=FakeGenerator, verifier=FakeVerifier())
    assert first == {"status": "shard_complete", "shard_index": 0,
                     "num_shards": 2, "questions": 1}
    assert second == {"status": "shard_complete", "shard_index": 1,
                      "num_shards": 2, "questions": 1}
    assert not (tmp_path / "summary.json").exists()
    assert len(FakeGenerator.calls) == 128 and len(set(FakeGenerator.calls)) == 128
    assert files_for(tmp_path, 0, 0)[1].is_file()
    assert files_for(tmp_path, 1, 0)[1].is_file()
    result = summarize(config, "base", items, tmp_path)
    assert result["questions"] == 2 and result["pass_at_k"]["1"] == 0.5


def test_overlap_scoring_preserves_outputs_and_resume(config, items, tmp_path):
    config["runtime"]["overlap_verification"] = True
    FakeGenerator.calls = []
    first = run(config, "base", items, tmp_path, generator_factory=FakeGenerator, verifier=FakeVerifier())
    assert first["total_samples"] == 128
    assert first["pass_at_k"]["1"] == 0.5
    assert first["pass_at_k"]["64"] == 1
    second = run(config, "base", items, tmp_path, generator_factory=FakeGenerator, verifier=FakeVerifier())
    assert second == first and len(FakeGenerator.calls) == 128


def test_overlap_failure_retains_generations_for_resume(config, items, tmp_path):
    config["runtime"]["overlap_verification"] = True
    class Interrupted(FakeVerifier):
        def verify(self, *args):
            raise RuntimeError("interrupted verification")
    FakeGenerator.calls = []
    with pytest.raises(RuntimeError, match="interrupted verification"):
        run(config, "base", items, tmp_path, generator_factory=FakeGenerator, verifier=Interrupted())
    assert 0 < len(FakeGenerator.calls) <= 128  # generated blocks persist despite verifier failure
    run(config, "base", items, tmp_path, generator_factory=FakeGenerator, verifier=FakeVerifier())
    assert len(FakeGenerator.calls) == 128


def test_interrupted_scoring_reuses_generated_samples(config,items,tmp_path):
    class Interrupted(FakeVerifier):
        def verify(self,*args):raise RuntimeError("interrupted")
    FakeGenerator.calls=[]
    with pytest.raises(RuntimeError,match="interrupted"):
        run(config,"base",items,tmp_path,generator_factory=FakeGenerator,verifier=Interrupted())
    assert len(FakeGenerator.calls)==16
    assert not (tmp_path/"summary.json").exists()
    run(config,"base",items,tmp_path,generator_factory=FakeGenerator,verifier=FakeVerifier())
    assert len(FakeGenerator.calls)==128


def test_missing_scores_cannot_produce_main_result(config,items,tmp_path):
    run(config,"base",items,tmp_path,generator_factory=FakeGenerator,verifier=FakeVerifier())
    files_for(tmp_path,1,56)[1].unlink()
    with pytest.raises(FileNotFoundError):summarize(config,"base",items,tmp_path)


def test_tampered_seed_and_scores_are_rejected(config,items,tmp_path):
    run(config,"base",items,tmp_path,generator_factory=FakeGenerator,verifier=FakeVerifier())
    generated,scored=files_for(tmp_path,0,0)
    value=json.loads(generated.read_text()); value["records"][0]["seed"]+=1
    atomic_json(generated,value)
    with pytest.raises(ValueError,match="IDs/seeds"):
        summarize(config,"base",items,tmp_path)


def test_truncation_counts_as_failure_and_infrastructure_errors_raise():
    verifier=FakeVerifier()
    assert score_sample(verifier,{}, {"text":"correct","finish_reason":"length"})["correct"] is False
    with pytest.raises(RuntimeError):score_sample(verifier,{}, {"text":"correct","finish_reason":"abort"})
    class Broken(FakeVerifier):
        def verify(self,*args):return SimpleNamespace(fully_correct=False,executable=False,error="worker_exit_1",metadata={})
    with pytest.raises(RuntimeError,match="infrastructure"):
        score_sample(Broken(),{}, {"text":"x","finish_reason":"stop"})


def test_seed_is_independent_of_batching_and_model_label():
    assert sample_seed(42,"q1",0)==sample_seed(42,"q1",0)
    assert len({sample_seed(42,q,n) for q in ("q1","q2") for n in range(64)})==128


def test_repeated_answers_are_not_deduplicated(config,items,tmp_path):
    class Constant(FakeGenerator):
        def generate(self,requests):
            return [{"text":"correct","finish_reason":"stop","response_tokens":1,"prompt_tokens":1} for _ in requests]
    result=run(config,"base",items,tmp_path,generator_factory=Constant,verifier=FakeVerifier())
    assert result["total_samples"]==128 and result["pass_at_k"]["1"]==1


def test_checkpoint_requires_real_weights_and_every_shard(tmp_path):
    model={"path":str(tmp_path),"checkpoint_marker":None}
    for n in ("config.json","tokenizer_config.json","tokenizer.json"):(tmp_path/n).write_text("{}")
    with pytest.raises(FileNotFoundError,match="No Hugging Face weights"):model_files(model)
    atomic_json(tmp_path/"model.safetensors.index.json",{"weight_map":{"a":"model-1.safetensors","b":"model-2.safetensors"}})
    (tmp_path/"model-1.safetensors").write_text("weights")
    with pytest.raises(FileNotFoundError):model_files(model)
    (tmp_path/"model-2.safetensors").write_text("weights")
    assert len(model_files(model))==6


def test_hf_snapshot_accepts_symlinked_shards_but_rejects_index_traversal(tmp_path):
    snapshot=tmp_path/"snapshot"; snapshot.mkdir()
    for n in ("config.json","tokenizer_config.json","tokenizer.json"):(snapshot/n).write_text("{}")
    blob=tmp_path/"blob"; blob.write_text("weights")
    (snapshot/"model-1.safetensors").symlink_to(blob)
    index=snapshot/"model.safetensors.index.json"
    atomic_json(index,{"weight_map":{"a":"model-1.safetensors"}})
    model={"path":str(snapshot),"checkpoint_marker":None}
    assert snapshot/"model-1.safetensors" in model_files(model)
    atomic_json(index,{"weight_map":{"a":"../blob"}})
    with pytest.raises(ValueError,match="Unsafe checkpoint"):
        model_files(model)


def test_incomplete_128_evaluation_cannot_report_64_as_complete(config,items,tmp_path):
    run(config,"base",items,tmp_path,generator_factory=FakeGenerator,verifier=FakeVerifier())
    config["sampling"].update(samples_per_question=128,k_values=[1,2,4,8,16,32,64,128])
    with pytest.raises(FileNotFoundError):
        summarize(config,"base",items,tmp_path)


def test_async_queue_persists_later_block_before_slow_first_block(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from evorl.evaluation.engine import VerificationQueue
    release=Event()
    class V(FakeVerifier):
        def verify(self,item,response):
            if response=='slow':
                assert release.wait(5), 'later block was trapped behind slow first block'
            return super().verify(item,response)
    with ThreadPoolExecutor(max_workers=2) as pool:
        q=VerificationQueue(pool,V(),2,tmp_path/'fail.json')
        slow={'problem_id':'a','sample_id':0,'seed':1,'text':'slow','finish_reason':'stop'}
        fast={**slow,'problem_id':'b','text':'correct'}
        q.submit(tmp_path/'slow.json',{'x':1},[slow],{})
        q.submit(tmp_path/'fast.json',{'x':2},[fast],{})
        q.drain(block=True)
        assert (tmp_path/'fast.json').exists()
        assert not (tmp_path/'slow.json').exists()
        assert q.peak<=2
        release.set();q.finish()
        assert (tmp_path/'slow.json').exists()
        assert q.completed==2

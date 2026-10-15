from elitereplay.evaluation.descriptors.structural_descriptor_v2 import features as f

def test_filter_merge():
    assert f('select a from t where a = (select max(a) from u)') == f('select a from t where a in (select max(a) from u)')

def test_aggregation():
    assert f('select a from t group by a')['aggregation']=='none'
    assert f('select a from t group by a having count(*)>1')['aggregation']=='grouped'
    assert f('select a from t group by a order by count(*)')['aggregation']=='grouped'
    assert f('select count(*) from t')['aggregation']=='global'

def test_joins():
    assert f('select a.x from a join b on a.x=b.x')['join']=='1+'
    assert f('select a.x from a join b on a.x=b.x join c on b.x=c.x')['join']=='1+'

def test_anti():
    assert f('select a.x from a left join b on a.x=b.x where b.x is null')['anti_join']=='left_join'
    assert f('select a.x from a left join b as r on a.x=r.x where r.x is null and a.x>1')['anti_join']=='left_join'
    assert f('select a.x from a left join b on a.x=b.x where not b.x is null')['anti_join']=='none'
    assert f('select a.x from a left join b on a.x=b.x where b.x is null or a.x>1')['anti_join']=='none'
    assert f('select a.x from a left join b on a.x=b.x where exists (select 1 from c as b where b.x is null)')['anti_join']=='none'
    assert f('select x from a where x not in (select x from b)')['anti_join']=='not_in'
    assert f('select x from a where x not in (1,2)')['anti_join']=='none'
    assert f('select x from a where not exists(select 1 from b where b.x=a.x)')['anti_join']=='not_exists'
    assert f('select x from a where exists(select 1 from b where b.x=a.x)')['anti_join']=='none'

def test_derived():
    assert f('select x from (select x from a) q')['subquery']=='derived'

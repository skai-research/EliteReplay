"""Versioned analysis descriptor. Independent of unchanged training archive cells."""
import sqlglot
from sqlglot import exp
from evorl.EliteReplay.subtypes import structural_subtype as old_subtype
from evorl.analysis.bird_modes import signature_id
VERSION='structural_coarsened_v2'

def old_features(sql):
    f=dict(x.split('=',1) for x in old_subtype('bird',sql).split(':',1)[1].split(';'))
    if f['join']=='many':f['join']='2+'
    return f

def unwrap(node):
    while isinstance(node,exp.Paren):node=node.this
    return node

def negated(node):
    parent=node.parent
    while isinstance(parent,exp.Paren):parent=parent.parent
    return isinstance(parent,exp.Not)

def conjuncts(node):
    node=unwrap(node)
    if isinstance(node,exp.And):
        yield from conjuncts(node.this);yield from conjuncts(node.expression)
    else:yield node

def features(sql):
    # Start from v1 so changes are restricted to the requested properties.
    f=old_features(sql)
    tree=sqlglot.parse_one(sql,read='sqlite')
    f['join']='0' if f['join']=='0' else '1+'
    if not tree.find(exp.AggFunc):f['aggregation']='none'
    subs=set(f['subquery'].split('+'))-{'none'}
    subs={'filter' if x in {'scalar','membership'} else x for x in subs}
    f['subquery']='+'.join(sorted(subs)) or 'none'
    anti=set()
    for node in tree.walk():
        if isinstance(node,exp.Exists) and negated(node):anti.add('not_exists')
        elif isinstance(node,exp.In) and node.args.get('query') is not None and negated(node):anti.add('not_in')
    # Only a positive conjunct in the SAME SELECT's WHERE qualifies. An IS NULL
    # inside OR, NOT, or a nested query must not turn this SELECT into an anti-join.
    for select in tree.find_all(exp.Select):
        right_aliases={j.this.alias_or_name.lower() for j in select.args.get('joins',[])
                       if j.side.upper()=='LEFT' and j.this.alias_or_name}
        where=select.args.get('where')
        if right_aliases and where:
            for predicate in conjuncts(where.this):
                if (isinstance(predicate,exp.Is) and isinstance(predicate.expression,exp.Null)
                    and isinstance(predicate.this,exp.Column)
                    and predicate.this.table.lower() in right_aliases):anti.add('left_join')
    f['anti_join']='+'.join(sorted(anti)) or 'none'
    return f

def readable(f):
    parts=[]
    if f['anti_join']!='none':
        names={'left_join':'LEFT JOIN','not_in':'NOT IN','not_exists':'NOT EXISTS'}
        parts.extend('anti-join ('+names[a]+')' for a in f['anti_join'].split('+'))
    if f['join']!='0':parts.append('1+ joins')
    if f['subquery']!='none':
        names={'filter':'filter subquery','exists':'EXISTS subquery','derived':'derived-table subquery'}
        parts.extend(names[x] for x in f['subquery'].split('+'))
    if f['aggregation']!='none':parts.append(f['aggregation']+' aggregation')
    if f['window']=='true':parts.append('window')
    if f['set']!='none':parts.append(f['set'])
    if f['topk']=='true':parts.append('top-k')
    return ' · '.join(parts) or 'plain selection'

def describe(sql):
    old=old_features(sql);new=features(sql)
    return {'old':signature_id(old),'new':signature_id(new),'old_features':old,'new_features':new}

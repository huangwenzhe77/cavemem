import argparse,base64,copy,hashlib,json,random,subprocess,sys
from pathlib import Path
from statistics import mean

D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parent/'cave-tb-pilot-01'))
import backend as b
sys.path.insert(0,str(D.parents[1]/'src'))
from memory_agent.cave.records import digest,read,save
from memory_agent.cave.alfworld_memory import paired_contrast
from remote import model_command, teacher_command
REG=read(D/'registration.json')
CONFIG={
    'polyglot-c-py':('c','/app/polyglot/main.py.c','ls -A /app'),
    'regex-log':('text','/app/regex.txt','ls -A /app'),
    'cancel-async-tasks':('python','/app/run.py','ls -A /app; python3 --version'),
    'regex-chess':('json','/app/re.json','ls -A /app; head -40 /app/check.py'),
    'log-summary-date-ranges':('python','/tmp/solution.py','ls /app/logs | head -12; f=$(ls /app/logs | head -1); head -5 "/app/logs/$f"'),
    'constraints-scheduling':('python','/tmp/solution.py','cat /app/*_calendar.ics | head -240')}
GEN={'constraints-scheduling','log-summary-date-ranges'}
def applies(c,t):
    return c=='artifact' or (c=='program' and (t in GEN or t in ('polyglot-c-py','cancel-async-tasks'))) or (c=='regex' and t in ('regex-log','regex-chess'))
def prompt(task,public,history,reminder=None,prior=None):
    lang,path,_=CONFIG[task]
    instruction=b.instruction(task)
    contract=('Return a complete Python program. It will be executed in /app to create the requested output file(s).'
              if task in GEN else 'Return ONLY the complete exact contents of '+path+'. The interface writes this file for you.')
    text=('Programming assignment\nTask: Write a file that contains the three-letter word cat.\n'
          'Answer (text file):\n```text\ncat\n```\n\nProgramming assignment\nTask: '+instruction+
          '\nInitial public directory/input preview:\n'+public+'\nSubmission interface: '+contract+
          ' Do not put explanations, Markdown or shell redirection inside the file. End the answer with a closing code fence on a new line.')
    if history:
        last=history[-1]
        text+='\nLatest submitted file content:\n'+last['draft']['body']+'\nPublic execution feedback:\n'+json.dumps(last['exec'],ensure_ascii=False)
        text+='\nReview that submission. If it fully meets the task, output just KEEP. Otherwise provide a complete replacement.'
    if reminder:
        text+='\nProposed answer awaiting execution:\n'+prior+'\nAdvisory from a previous game:\n'+reminder+'\nRevise the proposed answer only if useful.'
    return text+'\nAnswer ('+lang+' file):\n```'+lang+'\n'
class Actor:
    def __init__(self):
        self.p=subprocess.Popen(model_command(f"{D.name}/infer.py"),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf-8')
    def generate(self,prompt,seed):
        self.p.stdin.write(json.dumps({'prompt':prompt,'seed':seed})+'\n');self.p.stdin.flush()
        line=self.p.stdout.readline()
        if not line:raise RuntimeError('Qwen worker exited unexpectedly')
        d=json.loads(line);raw=d['text'].partition('\n```')[0].strip('\n')
        d.update(body=raw,seed=seed,prompt_hash=digest(prompt))
        return d
    def close(self):
        self.p.stdin.close();self.p.wait(timeout=60)
def seed_at(seed,step,kind):
    return None if seed is None else int(digest([seed,step,kind])[:12],16)
def apply(task,app,body):
    if body.strip()=='KEEP':return {'code':0,'stdout':'Kept current artifact.','stderr':'','seconds':0.}
    lang,path,_=CONFIG[task]
    body=(body+'\n').encode()
    if len(body)>140000:raise ValueError('artifact too large')
    folder=path.rsplit('/',1)[0]
    script=f"mkdir -p '{folder}' && printf %s '{base64.b64encode(body).decode()}' | base64 -d > '{path}'"
    if task in GEN:
        script+=" && python3 /tmp/solution.py && ls -l /app/"+('summary.csv' if task=='log-summary-date-ranges' else 'meeting_scheduled.ics')
    elif task=='cancel-async-tasks':script+=" && python3 -m py_compile /app/run.py"
    elif task=='polyglot-c-py':script+=' && gcc /app/polyglot/main.py.c -o /tmp/cmain && /tmp/cmain 4 && python3 /app/polyglot/main.py.c 4'
    elif task=='regex-chess':script+=" && python3 -m json.tool /app/re.json >/dev/null"
    return b.execute(task,app,script)
def totals(drafts,execs,reward):
    tokens=sum(d['input_tokens']+d['output_tokens'] for d in drafts)
    secs=sum(d['seconds'] for d in drafts)+sum(e['seconds'] for e in execs)
    steps=len(execs);prices=REG['cost']
    return {'success':reward,'tokens':tokens,'seconds':secs,'steps':steps,'actor_calls':len(drafts),
            'utility':reward-prices['per_1000_tokens']*tokens/1000-prices['per_step']*steps-prices['per_minute']*secs/60}
def rollout(task,w,path,source=None,step=1,arm='silent',candidate=None,seed=None):
    if (path/'result.json').exists():return read(path/'result.json')
    path.mkdir(parents=True,exist_ok=True); caps={}
    if source:
        pre=read(source/f'pre-{step}.json')
        app=b.clone(task,pre['snapshot'])
        assert b.state(task,app)==pre['state']
        rows=pre['history'];drafts=[r['draft'] for r in rows];execs=[r['exec'] for r in rows]
        public=pre['public'];original=pre['draft']
        for cap in REG['caps']:
            if cap<step:caps[str(cap)]=read(source/'result.json')['caps'][str(cap)]
    else:
        app=b.new(task);public=b.execute(task,app,CONFIG[task][2])['stdout'][:16000];rows=[];drafts=[];execs=[]
    for k in range(step,REG['limit']+1):
        if len(drafts)>=REG['limit']:break
        base=prompt(task,public,rows)
        if source and k==step:
            assert original['prompt_hash']==digest(base)
            d=original;drafts.append(original)
            if arm=='direct':d={'body':candidate['correction'],'kind':'teacher_direct'}
            if arm=='remind':
                if len(drafts) in REG['caps']:
                    # At a cap that funds only the untreated proposal, execute it under the common no-revision fallback.
                    clone=b.clone(task,app);e=apply(task,clone,original['body'])
                    g=b.grade(task,clone,path/f'cap-{len(drafts)}')
                    caps[str(len(drafts))]={'metrics':totals(drafts,[*execs,e],g['reward']),'grade':g,'fallback':True}
                d=w.generate(prompt(task,public,rows,candidate['reminder'],original['body']),seed_at(seed,k,'revision'));drafts.append(d)
        else:
            d=w.generate(base,seed_at(seed,k,'continuation'));drafts.append(d)
        if not source:
            clone=b.clone(task,app)
            save(path/f'pre-{k}.json',dict(snapshot=clone,state=b.state(task,clone),draft=d,public=public,history=rows))
        e=apply(task,app,d['body']);execs.append(e)
        rows.append({'step':k,'draft':d,'exec':e})
        save(path/f'turn-{k}.json',rows[-1])
        keep=d['body'].strip()=='KEEP'
        for cap in REG['caps']:
            if str(cap) not in caps and (len(drafts)==cap or (keep and cap>=len(drafts))):
                g=b.grade(task,app,path/f'cap-{cap}'); caps[str(cap)]={'metrics':totals(drafts,execs,g['reward']),'grade':g}
        if keep:break
    assert set(caps)=={'2','4'},(task,step,arm,caps)
    result={'task':task,'seed':seed,'arm':arm,'step':step,'rows':rows,'caps':caps,'state':b.state(task,app)}
    save(path/'result.json',result);print('SEALED',str(path.relative_to(D)),[(c,v['metrics']['success'],v['metrics']['tokens']) for c,v in caps.items()],flush=True)
    return result


def propose(task,trace):
    path=D/'source'/task/'proposal.json'
    if path.exists():return read(path)['candidate']
    data={'task':b.instruction(task),'format':CONFIG[task][:2],'mode':'Python generator' if task in GEN else 'complete artifact',
          'public_initial':read(D/'natural'/task/'pre-1.json')['public'],
          'steps':[{'step':r['step'],'draft':r['draft']['body'],'feedback':r['exec']} for r in trace['rows'] if r['step']<REG['limit']],
          'terminal_success':trace['caps']['4']['metrics']['success']}
    p=subprocess.run(
        teacher_command(f"{D.name}/teacher.py"),input=json.dumps(data),text=True,capture_output=True,encoding='utf-8',timeout=300)
    if p.returncode:raise RuntimeError(p.stderr[-700:])
    out=json.loads(p.stdout);out['source_digest']=digest(data)
    save(path,out);return out['candidate']

def verify(task,w):
    orig=D/'natural'/task
    base=read(orig/'result.json')
    if base['caps']['4']['metrics']['success']:return None
    folder=D/'source'/task
    if (folder/'evidence.json').exists():return read(folder/'evidence.json')
    c=propose(task,base);reps=[];direct_local=False
    if not applies(c['condition'],task):
        out={'source':task,'candidate':c,'admitted':False,'positive_direct':False,'reason':'condition_misses_source'}
        save(folder/'evidence.json',out);return out
    for k in REG['seeds']:
        order=['silent','direct','remind'];random.Random(k).shuffle(order)
        metrics={};first_effect={}
        for arm in order:
            res=rollout(task,w,folder/str(k)/arm,orig,c['step'],arm,c,k)
            metrics[arm]=res['caps']['4']['metrics']
            first_effect[arm]=res['rows'][c['step']-1]['exec']['code']
        direct_local |= first_effect['silent']!=0 and first_effect['direct']==0
        reps.append({'seed':k,'metrics':metrics,'public_check':first_effect})
    q={a:paired_contrast(reps,a) for a in ('direct','remind')}
    matrix=[(r['metrics']['silent']['success'],r['metrics']['remind']['success']) for r in reps]
    rescued=matrix.count((0,1));harmed=matrix.count((1,0))
    all_dual=all(p==(1,1) for p in matrix)
    direct_rescue=any(r['metrics']['silent']['success']<r['metrics']['direct']['success'] for r in reps)
    admitted=q['remind']['mean']>0 and (rescued>harmed or all_dual)
    out={'source':task,'candidate':c,'q':q,'rescued':rescued,'harmed':harmed,'all_dual_success':all_dual,
         'admitted':admitted,'positive_direct':direct_local or direct_rescue,'direct_local':direct_local,
         'direct_source_rescue':direct_rescue,'replicates':reps}
    save(folder/'evidence.json',out);print('SOURCE',task,admitted,matrix,flush=True);return out

def target(task,w,card,stratum):
    c=card['candidate'];path=D/'targets'/stratum/task/card['source'];src=D/'natural'/task
    rows={}
    for arm in sorted(('silent','remind'),key=lambda a:digest([a,task,card['source']])):
        rows[arm]=rollout(task,w,path/arm,src,1,arm,c,None)
    aa=read(src/'result.json')
    comparisons={}
    for k in ('2','4'):
        silent,remind=(rows[a]['caps'][k] for a in ('silent','remind'))
        assert silent['grade']['state']==aa['caps'][k]['grade']['state'] and silent['metrics']['success']==aa['caps'][k]['metrics']['success']
        comparisons[k]={'silent':silent['metrics'],'remind':remind['metrics'],
                        'delta':{m:remind['metrics'][m]-silent['metrics'][m] for m in silent['metrics']}}
    out={'task':task,'source':card['source'],'status':stratum,'comparisons':comparisons}
    save(path/'comparison.json',out);return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['teacher','finish'])
    args=parser.parse_args()
    if args.stage=='teacher':
        import time
        for t in REG['splits']['source']:
            p=D/'natural'/t/'result.json'
            while not p.exists():time.sleep(2)
            tr=read(p)
            if not tr['caps']['4']['metrics']['success']:propose(t,tr)
        return
    w=Actor()
    try:
        evidence=[v for t in REG['splits']['source'] if (v:=verify(t,w))]
        bank=[v for v in evidence if v['admitted']]
        save(D/'bank.json',bank)
        active=bank or [v for v in evidence if v['positive_direct']]
        pairs=[]
        for split in ('train','validation','test'):
            for t in REG['splits'][split]:
                for card in active:
                    if applies(card['candidate']['condition'],t):
                        pairs.append(target(t,w,card,'verified' if bank else 'direct_only_exploratory'))
        natural={t:read(D/'natural'/t/'result.json')['caps'] for ids in REG['splits'].values() for t in ids}
        report={'registration':digest(REG),'bank_size':len(bank),'source':evidence,'pairs':pairs,'natural':natural}
        save(D/'summary.json',report);print('COMPLETE',len(bank),len(pairs),flush=True)
    finally:w.close()

if __name__=='__main__':main()

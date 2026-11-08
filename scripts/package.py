"""Package only declared executables/DLLs and public licensing/provenance records."""
import hashlib,json,os,re,shutil,subprocess,sys,zipfile
from pathlib import Path
variant=sys.argv[1]; work=Path(sys.argv[2]); tools=Path(sys.argv[3]); build=work/'build'
release=tools/'release';release.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def run(args,cwd=None):
    p=subprocess.run(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print(p.stdout,flush=True)
    if p.returncode:raise RuntimeError(f'{args[0]} failed: {p.returncode}')
    return p.stdout
if variant=='finish':
    bundles={v:json.loads((release/(v+'-manifest.json')).read_text()) for v in ('candidate',)}
    total=sum(p.stat().st_size for p in release.iterdir() if p.is_file())
    assert total<700_000_000,'Compact bundle budget exceeded'
    write(release/'build-manifest.json',{'repository':os.environ['GITHUB_REPOSITORY'],'tools_commit':os.environ['GITHUB_SHA'],'run_id':os.environ['GITHUB_RUN_ID'],'run_attempt':os.environ['GITHUB_RUN_ATTEMPT'],'pins':json.loads((tools/'pins.json').read_text()),'toolchain':json.loads((work/'records/toolchain.json').read_text(encoding='utf-8-sig')),'bundles':bundles,'checks':'startup/help, CPU ADD, NAF CPU selftest; no Vulkan/model execution on runner'})
    (release/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(release.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt'),encoding='ascii')
    sys.exit(0)
assert variant == 'candidate'
out=work/('bundle-'+variant);out.mkdir()
expected=['trellis-cli.exe','trellis-server.exe','repair-stage.exe','repair-probe.exe','repair-conditioning.exe','trellis-test-naf.exe','ggml.dll','ggml-base.dll','ggml-cpu.dll','ggml-vulkan.dll']
for name in expected:
    found=list(build.rglob(name));assert len(found)==1,(name,found)
    shutil.copyfile(found[0],out/name)
notices=out/'notices';notices.mkdir()
for key,src in [('native',work/'native'),('ggml',work/'native/thirdparty/ggml'),('webp',work/'webp'),('spirv',work/'spirv')]:
    for p in src.rglob('*'):
        if not p.is_file() or '.git' in p.parts:continue
        rel=p.relative_to(src)
        if key=='native' and rel.parts[:2]==('thirdparty','ggml'):continue
        if re.search(r'license|copying|copyright|patents|authors|notice',p.name,re.I):
            dest=notices/key/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
# Single-file vendored libraries retain their embedded attribution in these public sources.
for folder in ['stb','xatlas','meshoptimizer','fqms','httplib']:
    src=work/'native/thirdparty'/folder
    if src.exists():
        for p in src.rglob('*'):
            if p.is_file() and p.suffix in ('.h','.hpp','.cpp','.c'):
                dest=notices/'native/thirdparty'/folder/p.relative_to(src);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
for name in ['NOTICE.md','LICENSE','pins.json']:shutil.copyfile(tools/name,out/name)
imports={}
for name in expected:
    imports[name]=run(['dumpbin','/DEPENDENTS',str(out/name)])
checks={}
checks['cli_help']=run([str(out/'trellis-cli.exe'),'--help'],out)
checks['stage_help']=run([str(out/'repair-stage.exe'),'--help'],out)
abi=run([str(out/'repair-probe.exe'),'--cpu'],out)
checks['cpu_probe']=abi
checks['naf_cpu_selftest']=run([str(out/'trellis-test-naf.exe'),'--selftest'],out)
checks['conditioning_help']=run([str(out/'repair-conditioning.exe'),'--help'],out)
checks['geometry_checkpoint']=run([str(out/'repair-conditioning.exe'),'geometry',str(work/'cpu-geometry-checkpoint')],out)
manifest={'variant':variant,'source_commit':json.loads((tools/'pins.json').read_text())['native']['commit'],'patches_applied':[] if variant=='baseline' else list(json.loads((tools/'pins.json').read_text())['patches']), 'tools_commit':os.environ['GITHUB_SHA'],'run_id':os.environ['GITHUB_RUN_ID'],'files':{str(p.relative_to(out)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()},'imports':imports,'cpu_probe_abi':abi,'configuration':json.loads((work/'records/configure-options.json').read_text(encoding='utf-8-sig')),'toolchain':json.loads((work/'records/toolchain.json').read_text(encoding='utf-8-sig'))}
manifest['startup_cpu_checks']={'exit_codes':{k:0 for k in checks},'output':checks}
manifest['naf_compile_command']=json.loads((work/'records/naf-compile-command.json').read_text(encoding='utf-8-sig'))
write(out/'bundle-manifest.json',manifest)
archive=release/(variant+'-windows-x64.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(out.rglob('*')):
        if p.is_file():z.write(p,str(p.relative_to(out)))
assert archive.stat().st_size<350_000_000
write(release/(variant+'-manifest.json'),{'name':archive.name,'sha256':sha(archive),'bytes':archive.stat().st_size,'bundle_manifest_sha256':sha(out/'bundle-manifest.json'),'runtime_files':{k:v for k,v in manifest['files'].items() if k in expected}})

#!/usr/bin/env python3
"""Read-only MP4/zip metadata audit. Не выводит generation latency из duration/имени."""
import argparse,hashlib,json,subprocess,tempfile,zipfile
from pathlib import Path


def inspect_video(path,name=None):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):digest.update(chunk)
    reply=subprocess.run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)],
                         check=True,capture_output=True,text=True,timeout=60)
    raw=json.loads(reply.stdout);tags=dict(raw.get('format',{}).get('tags',{}))
    for i,s in enumerate(raw.get('streams',[])):
        for k,v in s.get('tags',{}).items():tags[f'stream{i}:{k}']=v
    embedded={k:v for k,v in tags.items() if any(part in k.lower() for part in ('workflow','prompt','comment','description'))}
    return dict(file=name or path.name,sha256=digest.hexdigest(),size_bytes=path.stat().st_size,
        duration_s=float(raw['format']['duration']),streams=[{k:s.get(k) for k in
            ('codec_type','codec_name','width','height','nb_frames','avg_frame_rate','sample_rate','channels','duration')} for s in raw['streams']],
        tag_keys=sorted(tags),embedded_metadata=embedded or 'NOT_PRESENT',
        workflow='UNKNOWN' if not embedded else 'INSPECT_EMBEDDED_METADATA',seed='UNKNOWN',
        attention='UNKNOWN',sampler='UNKNOWN',generation_latency_s='UNKNOWN')


def main():
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.input);rows=[]
    if source.suffix.lower()=='.zip':
        with zipfile.ZipFile(source) as z,tempfile.TemporaryDirectory(prefix='powershard-media-') as tmp:
            for i,item in enumerate(z.infolist()):
                if item.is_dir() or Path(item.filename).suffix.lower() not in ('.mp4','.mov','.mkv','.webm'):continue
                # Никогда не использовать пользовательское archive path для записи.
                target=Path(tmp)/f'{i}{Path(item.filename).suffix}'
                with z.open(item) as src,target.open('wb') as dest:
                    for data in iter(lambda:src.read(2**20),b''):dest.write(data)
                rows.append(inspect_video(target,item.filename));target.unlink()
    elif source.is_dir():
        rows=[inspect_video(f) for f in sorted(source.glob('*.mp4'))]
    else:rows=[inspect_video(source)]
    seen={}
    for row in rows:
        row['duplicate_of']=seen.get(row['sha256']);seen.setdefault(row['sha256'],row['file'])
    report=dict(status='PASS',scope='container metadata only',files=len(rows),unique_files=len(seen),videos=rows,
        note_ru='Нет metadata — параметры UNKNOWN. Длительность видео не является временем генерации; артефакты нельзя приписать attention без исходных PNG и контрольного запуска.')
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(target)


if __name__=='__main__':main()

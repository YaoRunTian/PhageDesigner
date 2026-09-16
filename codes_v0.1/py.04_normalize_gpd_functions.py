#!/usr/bin/env python3
"""标准化 GPD 蛋白功能标签并生成类别统计。"""
from __future__ import annotations
import argparse, csv, json, logging, re, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/public8/lilab/student/rtyao/phage/phage_designer_v0.1')
DEFAULT_INPUT = PROJECT / 'results/03_gpd_qc/gpd_protein_manifest.tsv'
DEFAULT_OUT = PROJECT / 'results/04_gpd_function_labels'

PRODUCT_RULES = [
    ('host_interaction', re.compile(r'spike|tail fiber|receptor|adsorption|host range|injection|ejection', re.I)),
    ('structural', re.compile(r'capsid|coat protein|major tail|tail tube|portal|head protein|scaffold|baseplate|virion|structural|morphogenesis|assembly', re.I)),
    ('packaging', re.compile(r'packag|terminase|portal', re.I)),
    ('replication', re.compile(r'replicat|polymerase|primase|helicase|exonuclease|nuclease|recombin|dna-binding|restriction', re.I)),
    ('transcription', re.compile(r'transcription|transcriptase|rna polymerase|sigma factor|anti-termination', re.I)),
    ('translation', re.compile(r'translation|ribosomal|ribosome|trna|aminoacyl', re.I)),
    ('lysis', re.compile(r'lysis|lysin|holin|endolysin|spanin', re.I)),
    ('host_interaction', re.compile(r'host|receptor|adsorption|injection|integrase|inhibitor|anti-receptor', re.I)),
    ('defense', re.compile(r'anti.?crispr|defen[cs]|toxin|anti.?restriction|immunity', re.I)),
    ('metabolism', re.compile(r'metabol|dehydrogenase|kinase|synthetase|oxidoreductase|thioredoxin|ferredoxin', re.I)),
]
CLASSIFICATION_MAP = {
    'replication':'replication', 'transcription':'transcription',
    'translation':'translation', 'assembly':'structural',
    'packaging':'packaging', 'lysis':'lysis', 'infection':'host_interaction',
    'immune':'defense', 'integration':'host_interaction',
    'regulation':'transcription', 'metabolism':'metabolism',
}
UNKNOWN_RE = re.compile(r'(^|\b)(hypothetical|unknown|uncharacterized|unnamed|putative protein|protein of unknown function)(\b|$)', re.I)

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    p.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    p.add_argument('--dataset',default='GPD')
    return p.parse_args()

def logger_for(out):
    out.mkdir(parents=True,exist_ok=True)
    lg=logging.getLogger('py04'); lg.handlers.clear(); lg.setLevel(logging.INFO)
    fmt=logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    for h in (logging.StreamHandler(sys.stdout),logging.FileHandler(out/'run.log',mode='w',encoding='utf-8')):
        h.setFormatter(fmt); lg.addHandler(h)
    return lg

def normalize_product(product):
    text=(product or '').strip()
    if not text or UNKNOWN_RE.search(text): return 'unknown','unknown'
    for label, pattern in PRODUCT_RULES:
        if pattern.search(text): return label,'high'
    return 'unknown','weak'

def normalize_classification(classification):
    text=(classification or '').strip().lower()
    if not text or 'hypothetical' in text or 'unknown' in text or 'unsorted' in text:
        return 'unknown','unknown'
    for token in re.split(r'[;|,]', text):
        token=token.strip()
        if token in CLASSIFICATION_MAP: return CLASSIFICATION_MAP[token],'high'
    return 'unknown','weak'

def main():
    args=parse_args(); lg=logger_for(args.output_dir); start=time.time()
    if not args.input.exists(): lg.error('input does not exist: %s',args.input); return 2
    prefix=args.dataset.lower()
    out_manifest=args.output_dir/f'{prefix}_protein_function_manifest.tsv'
    counts=Counter(); raw_products=Counter(); raw_classes=Counter(); labels=Counter(); confidence=Counter(); split_labels=defaultdict(Counter)
    product_labels=Counter(); classification_labels=Counter(); product_confidence=Counter(); classification_confidence=Counter()
    fields=None
    with args.input.open('r',encoding='utf-8',newline='') as src, out_manifest.open('w',encoding='utf-8',newline='') as dst:
        reader=csv.DictReader(src,delimiter='\t')
        fields=reader.fieldnames or []
        output_fields=fields+['product_function','product_confidence','classification_function','classification_confidence','normalized_function','label_confidence']
        writer=csv.DictWriter(dst,fieldnames=output_fields,delimiter='\t',extrasaction='ignore'); writer.writeheader()
        for row in reader:
            counts['proteins']+=1
            product=(row.get('product') or '').strip(); classification=(row.get('classification') or '').strip()
            raw_products[product or '']+=1; raw_classes[classification or '']+=1
            plabel,pconf=normalize_product(product); clabel,cconf=normalize_classification(classification)
            label, conf = (plabel,pconf) if plabel != 'unknown' else (clabel,cconf)
            row['product_function']=plabel; row['product_confidence']=pconf
            row['classification_function']=clabel; row['classification_confidence']=cconf
            row['normalized_function']=label; row['label_confidence']=conf; writer.writerow(row)
            labels[label]+=1; confidence[conf]+=1; split_labels[row.get('split','')][label]+=1
            product_labels[plabel]+=1; classification_labels[clabel]+=1
            product_confidence[pconf]+=1; classification_confidence[cconf]+=1
            if counts['proteins']%500000==0: lg.info('processed %d proteins',counts['proteins'])
    summary={'dataset':args.dataset,'input':str(args.input),'generated_at':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':round(time.time()-start,2),'counts':dict(counts),'normalized_function_counts':dict(labels),'label_confidence_counts':dict(confidence),'product_function_counts':dict(product_labels),'classification_function_counts':dict(classification_labels),'product_confidence_counts':dict(product_confidence),'classification_confidence_counts':dict(classification_confidence),'split_function_counts':{k:dict(v) for k,v in split_labels.items()},'unique_raw_products':len(raw_products),'unique_raw_classifications':len(raw_classes),'top_raw_products':raw_products.most_common(50),'top_raw_classifications':raw_classes.most_common(50),'outputs':{'manifest':str(out_manifest)}}
    def write_counts(path, counter, field):
        with path.open('w', encoding='utf-8', newline='') as h:
            w=csv.writer(h, delimiter='\t'); w.writerow([field, 'count'])
            w.writerows(counter.most_common())
    product_counts_path=args.output_dir/f'{prefix.lower()}_product_counts.tsv'
    classification_counts_path=args.output_dir/f'{prefix.lower()}_classification_counts.tsv'
    write_counts(product_counts_path, raw_products, 'product')
    write_counts(classification_counts_path, raw_classes, 'classification')
    summary['all_raw_product_counts'] = dict(raw_products)
    summary['all_raw_classification_counts'] = dict(raw_classes)
    summary['outputs']['product_counts'] = str(product_counts_path)
    summary['outputs']['classification_counts'] = str(classification_counts_path)
    (args.output_dir/f'{prefix.lower()}_function_statistics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report=[f'# {args.dataset} protein function label normalization','',f"- Proteins processed: {counts['proteins']:,}",f"- Unique raw products: {len(raw_products):,}",f"- Unique raw classifications: {len(raw_classes):,}",'','| Normalized function | Count |','|---|---:|']
    report += [f'| {k} | {v:,} |' for k,v in labels.most_common()]
    report += ['','| Confidence | Count |','|---|---:|']+[f'| {k} | {v:,} |' for k,v in confidence.most_common()]
    report += ['','| Product-derived function | Count |','|---|---:|']+[f'| {k} | {v:,} |' for k,v in product_labels.most_common()]
    report += ['','| Classification-derived function | Count |','|---|---:|']+[f'| {k} | {v:,} |' for k,v in classification_labels.most_common()]
    report += ['','Rules preserve `product` and `classification`; normalized labels are intended for the first functional-alignment baseline, not definitive expert annotation.']
    (args.output_dir/f'{prefix.lower()}_function_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    lg.info('completed in %.1f minutes',(time.time()-start)/60); return 0
if __name__=='__main__': raise SystemExit(main())

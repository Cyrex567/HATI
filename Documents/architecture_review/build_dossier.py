"""Build the versioned HATI review PDF from Markdown and frozen evidence.

Uses saved GeoTIFFs/metrics only; never runs the scientific pipeline or ISIS.
Run with the scientific and document dependencies available on PYTHONPATH.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import re
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import ListedColormap
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE/'figures'
PDF = HERE/'HATI_architecture_and_implementation_review.pdf'
MD = HERE/'HATI_architecture_and_implementation_review.md'
REV = '40cf64075c331e19d7fb57c4b2b4b3b4de17be66'
GITHUB = 'https://github.com/Cyrex567/HATI/blob/'+REV+'/'
NAVY = '#142b43'; TEAL = '#276d70'; RED = '#b3223a'; GREY = '#718090'
CODE = {
 'C01': ('scripts/sweep_products.py', 'load_sweep'),
 'C02': ('scripts/ingest_sweep.py', 'coregister'),
 'C03': ('src/hati_core/landing_terrain.py', 'plane_metrics'),
 'C04': ('src/hati_core/dem_shadow.py', 'predict_visibility'),
 'C05': ('src/hati_core/shadow_likelihood.py', 'RegistrationProjector'),
 'C06': ('src/hati_core/regional_shadow.py', 'assess_regions'),
 'C07': ('src/hati_core/root_footprint.py', 'buffer_roots'),
 'C08': ('src/hati_core/scene_diagnostics.py', 'broad_dark_discrepancy'),
 'C09': ('scripts/landing_maps.py', 'run'),
 'C10': ('src/hati_core/adaptive_shadow.py', 'refine_cell'),
 'C11': ('scripts/adaptive_experiments.py', 't9'),
 'C12': ('src/hati_core/rock_scenes.py', 'render_rocks'),
 'C13': ('scripts/run_saturation_campaign.py', 'main'),
 'C14': ('dashboard/hati_watch.py', 'WatchStore'),
 'C15': ('tests/test_adaptive_shadow.py', None),
 'C16': ('src/hati_core/__init__.py', 'channels'),
}


def diagrams():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titleweight':'bold','savefig.facecolor':'white'})
    def canvas(size):
        fig, ax = plt.subplots(figsize=size)
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
        fig.subplots_adjust(left=.005,right=.995,top=.98,bottom=.015)
        return fig, ax
    def box(ax, x, y, w, h, label, fill='#edf2f6', dashed=False):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=GREY,lw=.9,linestyle='--' if dashed else '-'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9,color=NAVY,linespacing=1.35)
    def arrow(ax, a, b, style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,lw=1,color=TEAL,linestyle=style))
    fig, ax = canvas((6.9,3.3))
    box(ax,.05,4.72,4.2,1.1,'NAC EDRs + measured geometry\nISIS calibration and projection')
    box(ax,5.2,4.72,4.6,1.1,'Native DEM + physical configuration\nResolution and validity checks')
    box(ax,.05,2.85,4.2,1.2,'Aligned NAC stack + DEM visibility\nRegional moving-shadow search')
    box(ax,5.2,2.85,4.6,1.2,'Native fitted planes and relief\nTerrain index + DEM horizons')
    arrow(ax,(2.15,4.72),(2.15,4.05)); arrow(ax,(7.5,4.72),(7.5,4.05))
    arrow(ax,(5.2,3.65),(4.25,3.65)); ax.text(4.72,4.08,'visibility',ha='center',fontsize=8)
    box(ax,.05,1.03,3.1,1.2,'T9 adaptive context\nT10 mesh controls\nT11 prediction',fill='#f8f1e5',dashed=True)
    box(ax,4.05,1.03,5.75,1.2,'Exact root footprints + terrain buffer\nThree maps + qualification + attribution')
    arrow(ax,(1.6,2.85),(1.6,2.23),style='--')
    arrow(ax,(3.55,2.85),(4.95,2.23)); arrow(ax,(7.5,2.85),(7.5,2.23))
    ax.text(1.6,.72,'Experimental diagnostics',ha='center',fontsize=8,color=RED)
    ax.text(6.95,.55,'Sequential runner, read-only Watch and verified result ZIP',ha='center',fontsize=8.5)
    arrow(ax,(6.95,1.03),(6.95,.76))
    fig.savefig(FIG/'architecture.png',dpi=260); plt.close(fig)
    fig, ax = canvas((6.9,2.2))
    box(ax,.1,3.7,2.5,1.7,'Baseline assessed\nWarning or cutoff?')
    box(ax,3.4,3.7,2.8,1.7,'Actual larger patch\nFixed frame set\nTerrain guard')
    box(ax,7,3.7,2.8,1.7,'Fit both hypotheses\nRefine dimensions\nCheck endpoint support')
    arrow(ax,(2.6,4.55),(3.4,4.55)); arrow(ax,(6.2,4.55),(7,4.55))
    box(ax,.1,.55,3.2,1.75,'Unsupported or failed\nUnresolved\nNo successful fallback',fill='#f8eeee')
    box(ax,3.85,.55,2.6,1.75,'Context insufficient\nExpand 1x to 2x to 4x',fill='#f8f1e5')
    box(ax,7,.55,2.8,1.75,'Stop with explicit state\nLow evidence or\ncontext supported*',fill='#edf5f2')
    arrow(ax,(3.7,3.7),(2.2,2.3)); arrow(ax,(7.3,3.7),(5.6,2.3)); arrow(ax,(8.4,3.7),(8.4,2.3))
    arrow(ax,(4.75,2.3),(4.75,3.7),style='--')
    ax.text(8.4,.15,'*Unvalidated',ha='center',fontsize=8,color=RED)
    fig.savefig(FIG/'adaptive_flow.png',dpi=260); plt.close(fig)


def measured_figures(evidence, raster_dir):
    import rasterio
    run=evidence['run']; cf=evidence['counterfactual']; px=run['image_posting_m']
    r,c=cf['row_px'],cf['col_px']
    fig,axes=plt.subplots(2,2,figsize=(6.9,5.5),layout='constrained')
    fields=[('terrain_hazard','Terrain index'),('shadow_hazard','Shadow index'),('fused_hazard','Fused index'),('fusion_status','Fusion observability')]
    cmap=plt.get_cmap('YlOrRd').with_extremes(bad='#8795a4')
    for ax,(name,title) in zip(axes.flat,fields):
        with rasterio.open(raster_dir/(name+'.tif')) as src:
            a=src.read(1)
        h,w=a.shape; extent=(-c*px,(w-c)*px,(r-h)*px,r*px)
        if name=='fusion_status':
            im=ax.imshow(a,extent=extent,vmin=-.5,vmax=2.5,cmap=ListedColormap(['#8795a4','#438e86','#e0a155']),interpolation='nearest')
            bar=fig.colorbar(im,ax=ax,shrink=.78,ticks=[0,1,2]);bar.ax.set_yticklabels(['Unknown','Both qualified','High incomplete'],fontsize=7)
        else:
            im=ax.imshow(a,extent=extent,vmin=0,vmax=1,cmap=cmap,interpolation='nearest')
            fig.colorbar(im,ax=ax,shrink=.78,ticks=[0,.5,1])
        ax.scatter([0],[0],marker='+',s=70,color='#152b43',linewidths=1.4)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Map east offset (m)',fontsize=8);ax.set_ylabel('Map north offset (m)',fontsize=8)
        ax.tick_params(labelsize=7)
    fig.savefig(FIG/'athena_maps.png',dpi=270);plt.close(fig)
    rows=evidence['review_metrics']['controls']; fig,ax=plt.subplots(figsize=(6.9,2.95),layout='constrained')
    labels=['Static background','Changing background','Resolved ridge','0.3 m caster','0.6 m caster','1.2 m caster']
    values=[r['warnings']/r['assessed']*100 for r in rows]
    bars=ax.barh(np.arange(len(rows)),values,color=[TEAL,RED,GREY,TEAL,TEAL,TEAL],height=.6)
    ax.set_yticks(range(len(rows)),labels,fontsize=9);ax.invert_yaxis();ax.set_xlim(0,121)
    for bar,row,value in zip(bars,rows,values):
        ax.text(value+1.5,bar.get_y()+bar.get_height()/2,f"{row['warnings']}/{row['assessed']}",va='center',fontsize=9)
    ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Assessable trials containing any warning (%)',fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title('Previous workstation campaign at commit 8347e6d',fontsize=10,pad=10)
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    fig.savefig(FIG/'control_results.png',dpi=270);plt.close(fig)


def fonts():
    folder=Path('C:/Windows/Fonts')
    choices=[('Body','calibri.ttf'),('BodyB','calibrib.ttf'),('BodyI','calibrii.ttf'),('Mono','consola.ttf')]
    if not all((folder/f).exists() for _,f in choices):
        from matplotlib.font_manager import findfont
        choices=[('Body',findfont(FontProperties(family='DejaVu Sans'))),('BodyB',findfont(FontProperties(family='DejaVu Sans',weight='bold'))),('BodyI',findfont(FontProperties(family='DejaVu Sans',style='italic'))),('Mono',findfont(FontProperties(family='DejaVu Sans Mono')))]
        folder=Path('.')
    for name,file in choices:pdfmetrics.registerFont(TTFont(name,str(folder/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyB',italic='BodyI',boldItalic='BodyB')


def code_links(evidence):
    rows=[]
    for key,(path,symbol) in CODE.items():
        line=evidence['sources'].get(path,{}).get('symbols',{}).get(symbol)
        url=GITHUB+path+(f'#L{line}' if line else '')
        label=path.rsplit('/',1)[-1]+(' - '+symbol if symbol else '')
        rows.append(f'**{key}** [{label}]({url})')
    return '\n\n'.join(rows)


def inline(s):
    # Parse links before escaping so URLs and angle brackets remain valid.
    tokens={}
    def token(value):
        key=f'ZZTOKEN{len(tokens)}ZZ';tokens[key]=value;return key
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:token(f'<link href="{html.escape(m[2],quote=True)}" color="#245979">{html.escape(m[1])}</link>'),s)
    s=html.escape(s)
    s=re.sub(r'`([^`]+)`',lambda m:'<font name="Mono" size="9.2">'+m[1]+'</font>',s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    for key,value in tokens.items():s=s.replace(key,value)
    return s


class Dossier(BaseDocTemplate):
    def __init__(self,filename):
        self.heading_pages=[]
        super().__init__(filename,pagesize=A4,leftMargin=47,rightMargin=47,topMargin=49,bottomMargin=43,
                         title='HATI architecture and implementation review',author='HATI',
                         subject='Scientific and engineering review of deterministic lunar hazard sensing')
        self.addPageTemplates(PageTemplate(id='all',frames=[Frame(47,43,A4[0]-94,A4[1]-92,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=self.page))
    def page(self,canvas,doc):
        canvas.saveState()
        if doc.page>1:
            canvas.setFont('Body',8);canvas.setFillColor(colors.black)
            canvas.drawString(47,A4[1]-29,'HATI  |  Architecture and implementation review')
            canvas.drawRightString(A4[0]-47,A4[1]-29,'24 September 2026')
        canvas.setFont('Body',8);canvas.setFillColor(colors.HexColor('#56616d'))
        canvas.drawString(47,25,'Research system  |  Implementation 40cf640')
        canvas.drawRightString(A4[0]-47,25,str(doc.page))
        canvas.restoreState()
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and getattr(flowable,'is_heading',False):
            text=flowable.getPlainText(); key='s'+str(len(self.heading_pages))
            self.canv.bookmarkPage(key);self.canv.addOutlineEntry(text,key,level=0)
            self.heading_pages.append({'heading':text,'page':self.page})


def build(evidence):
    fonts();FIG.mkdir(exist_ok=True)
    styles={
      'body':ParagraphStyle('body',fontName='Body',fontSize=11,leading=14.35,spaceAfter=8,textColor=colors.HexColor('#19232d')),
      'question':ParagraphStyle('question',fontName='Body',fontSize=10.5,leading=13.2,spaceAfter=7,leftIndent=17,bulletIndent=0,textColor=colors.HexColor('#19232d')),
      'h1':ParagraphStyle('h1',fontName='BodyB',fontSize=19,leading=22,spaceAfter=14,textColor=colors.black,keepWithNext=True),
      'h2':ParagraphStyle('h2',fontName='BodyB',fontSize=13.5,leading=17,spaceBefore=7,spaceAfter=11,textColor=colors.black,keepWithNext=True),
      'title':ParagraphStyle('title',fontName='BodyB',fontSize=27,leading=31,spaceAfter=13,textColor=colors.black),
      'caption':ParagraphStyle('caption',fontName='BodyI',fontSize=9,leading=11.7,spaceBefore=5,spaceAfter=10,textColor=colors.HexColor('#465360')),
      'table':ParagraphStyle('table',fontName='Body',fontSize=9.4,leading=11.5,spaceAfter=0,textColor=colors.black),
      'th':ParagraphStyle('th',fontName='BodyB',fontSize=9.4,leading=11.5,textColor=colors.white),
      'code':ParagraphStyle('code',fontName='Mono',fontSize=8,leading=10.5,spaceAfter=9,textColor=colors.black),
      'reference':ParagraphStyle('reference',fontName='Body',fontSize=9.3,leading=11.6,spaceAfter=5,textColor=colors.HexColor('#19232d')),
    }
    text=MD.read_text(encoding='utf-8').replace('<!-- CODE_LINKS -->',code_links(evidence))
    # Expanded Markdown is also immediately usable without builder directives.
    (HERE/'HATI_review_for_feedback.md').write_text(text,encoding='utf-8')
    pages=text.split('<!-- PAGE -->');story=[];eq=0
    width=A4[0]-94
    for page_index,page in enumerate(pages):
        if page_index:story.append(PageBreak())
        if page_index==0:
            story.extend([Spacer(1,6),Image(str(FIG/'hati-logo.png'),width=165,height=165),Spacer(1,19)])
        lines=page.strip().splitlines();i=0
        reference_page=page.startswith('\n\n# 29')
        while i<len(lines):
            line=lines[i].strip()
            if not line:i+=1;continue
            if line.startswith('```'):
                kind=line[3:];body=[];i+=1
                while i<len(lines) and not lines[i].startswith('```'):
                    body.append(lines[i]);i+=1
                i+=1
                if kind=='math':
                    eq+=1;path=FIG/f'equation_{eq:02d}.png';latex=' '.join(body).replace('\\hbox','\\mathrm')
                    mathtext.math_to_image('$'+latex+'$',str(path),prop=FontProperties(family='DejaVu Serif',size=14),dpi=260,color='black')
                    with PILImage.open(path) as im:w,h=im.size
                    target=min(width,w*72/260);story.extend([Spacer(1,5),Image(str(path),width=target,height=h*target/w),Spacer(1,11)])
                else:
                    wrapped=[]
                    for raw in body:
                        # Presentation wrapping is visual only; copyable Markdown retains the exact command.
                        chunks=textwrap.wrap(raw,width=91,subsequent_indent='    ',break_long_words=True,break_on_hyphens=False) or ['']
                        wrapped.extend(chunks)
                    story.append(Paragraph('<br/>'.join(html.escape(x).replace(' ','&#160;') for x in wrapped),styles['code']))
                continue
            if line.startswith('# '):
                p=Paragraph(inline(line[2:]),styles['title' if page_index==0 else 'h1']);p.is_heading=True;story.append(p);i+=1;continue
            if line.startswith('## '):
                story.append(Paragraph(inline(line[3:]),styles['h2']));i+=1;continue
            match=re.match(r'!\[([^]]+)\]\(([^)]+)\)',line)
            if match:
                path=HERE/match[2]
                with PILImage.open(path) as im:w,h=im.size
                target=width;maxh=290 if 'athena_maps' in str(path) else 245
                if target*h/w>maxh:target=maxh*w/h
                story.append(Image(str(path),width=target,height=target*h/w))
                captions={'architecture.png':'Figure 1. Current data flow. Dashed branch: experimental outputs, separate from baseline fusion.',
                          'adaptive_flow.png':'Figure 2. Adaptive control flow. Final-pass failure remains unresolved; earlier success is not substituted.',
                          'athena_maps.png':'Figure 3. Recorded Athena indices and observability. Cross marks the fixed test coordinate. Grey is unavailable; indices are not probabilities. Source E1.',
                          'control_results.png':'Figure 4. Previous control responses. Each scenario had 72 planned trials and 56 assessable trials. Ridge warnings are not necessarily false hazards. Source E1.'}
                story.append(Paragraph(captions.get(path.name,match[1]),styles['caption']));i+=1;continue
            if line.startswith('|'):
                rows=[]
                while i<len(lines) and lines[i].strip().startswith('|'):
                    cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                    if not all(re.fullmatch(r'[:\- ]+',c) for c in cells):rows.append(cells)
                    i+=1
                n=len(rows[0]);
                if n==2: widths=[width*.23,width*.77]
                elif n==3:widths=[width*.28,width*.34,width*.38]
                else:widths=[width*.34]+[width*.66/(n-1)]*(n-1)
                if page_index in (10,):widths=[width*.18,width*.82]
                if page_index == 1:widths=[width*.75,width*.25]
                if page_index in (18,27):widths=[width*.12,width*.88]
                data=[[Paragraph(inline(c),styles['th' if r==0 else 'table']) for c in row] for r,row in enumerate(rows)]
                t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f1f4f7')]),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#d9d9d9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
                story.extend([t,Spacer(1,12)]);continue
            parts=[line];i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','![')) and not re.match(r'^\d+\. ',lines[i].strip()):
                parts.append(lines[i].strip());i+=1
            body=' '.join(parts)
            style=styles['reference'] if reference_page else styles['body']
            if re.match(r'^\d+\. ',body):
                num,body=body.split('. ',1);story.append(Paragraph(inline(body),styles['question'],bulletText=num+'.'))
            else:story.append(Paragraph(inline(body),style))
    doc=Dossier(str(PDF));doc.build(story)
    qa_dir=ROOT/'tmp/architecture_review'
    qa_dir.mkdir(parents=True,exist_ok=True)
    (qa_dir/'layout_index.json').write_text(json.dumps(doc.heading_pages,indent=2)+'\n',encoding='utf-8')
    print('PDF:',PDF)
    print('Heading page count:',len(doc.heading_pages),'last heading page:',doc.heading_pages[-1]['page'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raster-dir',type=Path,default=ROOT/'tmp/architecture_review')
    ap.add_argument('--reuse-figures',action='store_true');args=ap.parse_args()
    evidence=json.loads((HERE/'evidence_snapshot.json').read_text(encoding='utf-8'))
    FIG.mkdir(exist_ok=True)
    if not args.reuse_figures:diagrams();measured_figures(evidence,args.raster_dir)
    build(evidence)


if __name__=='__main__':main()

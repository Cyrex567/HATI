"""Categorical LROC NAC DTM confidence, per the NOBILE03 product README."""
import numpy as np

SOURCE='https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/NOBILE03/NAC_DTM_NOBILE03_README.TXT'
LABELS={0:'NoData / outside boundary',1:'Shadowed',2:'Saturated',3:'Suspicious',
        4:'Interpolated / extrapolated',**{i:'Successful correlation' for i in range(10,15)},15:'Manually edited'}


def category_counts(values):
    a=np.asarray(values)
    codes,counts=np.unique(a[np.isfinite(a)],return_counts=True)
    return [{'code':float(k),'description':LABELS.get(k,'Undocumented code'),
             'count':int(n)} for k,n in zip(codes,counts)]


def sample_categories(values,row,col,radius=2):
    a=np.asarray(values)
    if a.ndim!=2 or not 0<=row<a.shape[0] or not 0<=col<a.shape[1] or radius<0:
        raise ValueError('confidence sample outside raster or invalid radius')
    value=float(a[row,col])
    window=a[max(0,row-radius):row+radius+1,max(0,col-radius):col+radius+1]
    return dict(row=int(row),col=int(col),code=value if np.isfinite(value) else None,
                description=LABELS.get(value,'Undocumented or missing code'),
                window_shape=list(window.shape),categories=category_counts(window))

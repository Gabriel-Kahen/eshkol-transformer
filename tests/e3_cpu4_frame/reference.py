"""Independent controlled-parameter binary64 whole-model oracle, development only."""
import math
from pathlib import Path
import struct
import sys

SHAPES=((4,4),(4,4),(4,4),(4,4),(4,8),(8,4),(4,),(4,),
        (4,),(4,),(256,4),(4,),(4,),(4,4))
PARAMETERS=[]
for index,shape in enumerate(SHAPES):
    count=math.prod(shape)
    PARAMETERS.append([((i*17+index*11)%31-15)/64 for i in range(count)])
    if index in (7,9,12): PARAMETERS[-1]=[1+x/4 for x in PARAMETERS[-1]]

def linear(x,w,width,out):
    return [sum(x[t*width+d]*w[o*width+d] for d in range(width))
            for t in range(4) for o in range(out)]
def norm(x,weight,bias):
    out=[]
    epsilon=struct.unpack('<f',struct.pack('<I',0x3727c5ac))[0]
    for t in range(4):
        row=x[t*4:t*4+4];mean=sum(row)/4
        variance=sum((v-mean)**2 for v in row)/4
        out.extend((row[d]-mean)/math.sqrt(variance+epsilon)*weight[d]+bias[d]
                   for d in range(4))
    return out

def trace(ids):
    p=PARAMETERS
    et=[p[10][i*4+d] for i in ids for d in range(4)]
    ep=p[13][:];x=[a+b for a,b in zip(et,ep)]
    n1=norm(x,p[7],p[6]);q,k,v=[linear(n1,p[i],4,4) for i in (2,0,3)]
    split=lambda a:[a[t*4+h*2+d] for h in range(2) for t in range(4) for d in range(2)]
    qh,kh,vh=map(split,(q,k,v));ah=[]
    for h in range(2):
        for t in range(4):
            scores=[sum(qh[h*8+t*2+d]*kh[h*8+s*2+d] for d in range(2))/math.sqrt(2)
                    for s in range(t+1)]
            m=max(scores);w=[math.exp(a-m) for a in scores];total=sum(w)
            ah.extend(sum(w[s]/total*vh[h*8+s*2+d] for s in range(t+1)) for d in range(2))
    at=[ah[h*8+t*2+d] for t in range(4) for h in range(2) for d in range(2)]
    ao=linear(at,p[1],4,4);res=[a+b for a,b in zip(x,ao)]
    n2=norm(res,p[9],p[8]);fu=linear(n2,p[5],4,8)
    fg=[0.5*a*(1+math.erf(a/math.sqrt(2))) for a in fu]
    fd=linear(fg,p[4],8,4);y=[a+b for a,b in zip(res,fd)]
    nf=norm(y,p[12],p[11]);z=linear(nf,p[10],4,256)
    return (et,ep,x,n1,q,k,v,qh,kh,vh,ah,at,ao,res,n2,fu,fg,fd,y,nf,z)

IDS=(3,197,3,41)
TARGETS=(197,3,41,7)
MASK=(1,1,1,0)

def main(path):
    rows=trace(IDS)
    text='/* Independently generated controlled finite parameter/role values. */\n'
    for i,values in enumerate(PARAMETERS):
        words=[struct.unpack('<I',struct.pack('<f',v))[0] for v in values]
        text+=f'static const uint32_t parameter_{i}[]={{'+','.join(hex(w)+'u' for w in words)+'};\n'
    text+='static const uint32_t *const parameter_bits[]={'+','.join(f'parameter_{i}' for i in range(14))+'};\n'
    text+='static const size_t parameter_counts[]={'+','.join(str(len(p)) for p in PARAMETERS)+'};\n'
    for i,values in enumerate(rows):
        text+=f'static const double role_{i}[]={{'+','.join(v.hex() for v in values)+'};\n'
    text+='static const double *const expected_roles[]={'+','.join(f'role_{i}' for i in range(21))+'};\n'
    text+='static const size_t role_counts[]={'+','.join(str(len(v)) for v in rows)+'};\n'
    z=rows[-1];ce=[];correct=0
    for t,target in enumerate(TARGETS):
        row=z[t*256:(t+1)*256];m=max(row)
        ce.append(m+math.log(sum(math.exp(v-m) for v in row))-row[target])
        correct+=MASK[t]*(max(range(256),key=lambda j:row[j])==target)
    mean=sum(a*b for a,b in zip(ce,MASK))/sum(MASK)
    text+='static const double expected_ce[]={'+','.join(v.hex() for v in ce)+'};\n'
    text+='static const double expected_metrics[]={'+','.join(v.hex() for v in (mean,float(sum(MASK)),math.exp(mean),correct/sum(MASK)))+'};\n'
    Path(path).write_text(text)

if __name__=='__main__':main(sys.argv[1])

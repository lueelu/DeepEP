// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

// Dependency-free rectilinear routing for the offline SVG. Every connector has
// explicit source/target ports, orthogonal segments, and a non-overlapping label.
const key=(x,y)=>x+','+y;
const labelWidth=text=>Math.ceil([...text].reduce((s,c)=>s+(c.charCodeAt(0)>255?10.8:5.9),0))+18;
const overlaps=(a,b,pad=0)=>a.x<b.x+b.w+pad&&a.x+a.w+pad>b.x&&a.y<b.y+b.h+pad&&a.y+a.h+pad>b.y;
function sides(a,b,reverse){
 if(reverse)return ['bottom','bottom'];
 const dx=b.x+b.w/2-a.x-a.w/2,dy=b.y+b.h/2-a.y-a.h/2;
 if(Math.abs(dy)<100&&Math.abs(dx)>100)return dx>0?['right','left']:['left','right'];
 return dy>=0?['bottom','top']:['top','bottom'];
}
function port(n,side,t){
 const p=side==='left'?{x:n.x,y:n.y+n.h*t}:side==='right'?{x:n.x+n.w,y:n.y+n.h*t}:side==='top'?{x:n.x+n.w*t,y:n.y}:{x:n.x+n.w*t,y:n.y+n.h};
 const q={...p};q[side==='left'||side==='right'?'x':'y']+=side==='left'||side==='top'?-24:24;
 return [p,q];
}
function simplify(points){const r=[];for(const p of points){if(r.length&&r.at(-1).x===p.x&&r.at(-1).y===p.y)continue;while(r.length>=2){const a=r.at(-2),b=r.at(-1);if(a.x===b.x&&b.x===p.x||a.y===b.y&&b.y===p.y)r.pop();else break;}r.push(p);}return r;}
function clear(a,b,rects){
 return !rects.some(r=>a.x===b.x?a.x>r.x&&a.x<r.x+r.w&&Math.min(a.y,b.y)<r.y+r.h&&Math.max(a.y,b.y)>r.y:a.y>r.y&&a.y<r.y+r.h&&Math.min(a.x,b.x)<r.x+r.w&&Math.max(a.x,b.x)>r.x);
}
class Heap{
 q=[];
 push(v){let i=this.q.length;this.q.push(v);while(i){const p=(i-1)>>1;if(this.q[p].f<=v.f)break;this.q[i]=this.q[p];i=p;}this.q[i]=v;}
 pop(){const a=this.q[0],v=this.q.pop();if(this.q.length){let i=0;while(i*2+1<this.q.length){let c=i*2+1;if(c+1<this.q.length&&this.q[c+1].f<this.q[c].f)c++;if(this.q[c].f>=v.f)break;this.q[i]=this.q[c];i=c;}this.q[i]=v;}return a;}
}
function search(start,end,nodes,width,height,used){
 const rects=nodes.map(n=>({x:n.x-12,y:n.y-12,w:n.w+24,h:n.h+24}));
 const xs=[10,width-10,start.x,end.x],ys=[52,height-10,start.y,end.y];
 for(const n of nodes){xs.push(n.x-28,n.x+n.w+28);ys.push(n.y-28,n.y+n.h+28);}
 const X=[...new Set(xs.filter(v=>v>=8&&v<=width-8))].sort((a,b)=>a-b),Y=[...new Set(ys.filter(v=>v>=40&&v<=height-8))].sort((a,b)=>a-b);
 const sx=X.indexOf(start.x),sy=Y.indexOf(start.y),ex=X.indexOf(end.x),ey=Y.indexOf(end.y);
 if([sx,sy,ex,ey].some(v=>v<0))throw Error('Router port outside viewport');
 const nx=X.length,N=nx*Y.length,encode=(x,y,d)=>(y*nx+x)*3+d;
 const dist=new Float64Array(N*3).fill(Infinity),parent=new Int32Array(N*3).fill(-1),h=new Heap(),first=encode(sx,sy,2);
 const target=key(end.x,end.y);dist[first]=0;h.push({id:first,g:0,f:Math.abs(start.x-end.x)+Math.abs(start.y-end.y)});
 let goal=-1;
 while(h.q.length){const v=h.pop();if(v.g!==dist[v.id])continue;const d=v.id%3,i=Math.floor(v.id/3),x=i%nx,y=Math.floor(i/nx),a={x:X[x],y:Y[y]};if(key(a.x,a.y)===target){goal=v.id;break;}
  for(const [dx,dy,nd]of [[-1,0,0],[1,0,0],[0,-1,1],[0,1,1]]){
   const xx=x+dx,yy=y+dy;if(xx<0||xx>=nx||yy<0||yy>=Y.length)continue;
   const b={x:X[xx],y:Y[yy]};if(!clear(a,b,rects))continue;
   const len=Math.abs(a.x-b.x)+Math.abs(a.y-b.y),cost=len+(d!==2&&d!==nd?28:0)+(used.get([key(a.x,a.y),key(b.x,b.y)].sort().join('|'))||0)*18;
   const id=encode(xx,yy,nd),g=v.g+cost;if(g>=dist[id])continue;dist[id]=g;parent[id]=v.id;h.push({id,g,f:g+Math.abs(b.x-end.x)+Math.abs(b.y-end.y)});
  }
 }
 if(goal<0)throw Error('No obstacle-free connector: '+key(start.x,start.y)+' -> '+target);
 const r=[];for(let id=goal;id>=0;id=parent[id]){const i=Math.floor(id/3);r.push({x:X[i%nx],y:Y[Math.floor(i/nx)]});}r.reverse();
 for(let i=1;i<r.length;i++){const k=[key(r[i-1].x,r[i-1].y),key(r[i].x,r[i].y)].sort().join('|');used.set(k,(used.get(k)||0)+1);}
 return simplify(r);
}
function labelBox(points,text,nodes,labels,width){
 const w=labelWidth(text),h=24,segments=[];
 for(let i=1;i<points.length;i++){const a=points[i-1],b=points[i],horizontal=a.y===b.y,len=Math.abs(a.x-b.x)+Math.abs(a.y-b.y);segments.push({a,b,horizontal,len});}
 segments.sort((a,b)=>(b.horizontal?1000:0)+b.len-((a.horizontal?1000:0)+a.len));
 for(const s of segments){if(s.horizontal&&s.len<w+8)continue;
  for(const t of [.5,.2,.8,.1,.9]){
   const x=s.a.x+(s.b.x-s.a.x)*t,y=s.a.y+(s.b.y-s.a.y)*t;
   const candidates=s.horizontal?[{x:x-w/2,y:y-32,w,h},{x:x-w/2,y:y+8,w,h}]:[{x:x+10,y:y-h/2,w,h},{x:x-w-10,y:y-h/2,w,h}];
   for(const r of candidates)if(r.x>=0&&r.x+r.w<=width&&!nodes.some(n=>overlaps(r,n,5))&&!labels.some(n=>overlaps(r,n,5)))return r;
  }
 }
 throw Error('No clear label placement: '+text);
}
export function routeGraph(g){
 const lookup=Object.fromEntries(g.nodes.map(n=>[n.id,n])),ports=new Map(),labels=[],used=new Map();
 g.edges.forEach((e,i)=>{e.id=g.id+'-edge-'+i;const a=lookup[e.from],b=lookup[e.to],reverse=g.edges.some(q=>q.from===e.to&&q.to===e.from)&&a.x>b.x;[e.sourceSide,e.targetSide]=sides(a,b,reverse);const gap=Math.max(b.x-a.x-a.w,a.x-b.x-b.w);if(!reverse&&e.label&&['left','right'].includes(e.sourceSide)&&gap<labelWidth(e.label)+20){e.sourceSide='top';e.targetSide='top';}for(const [node,side,end]of [[e.from,e.sourceSide,'s'],[e.to,e.targetSide,'t']]){const k=node+':'+side;const a=ports.get(k)||[];a.push({e,end});ports.set(k,a);}});
 for(const list of ports.values())list.forEach((v,i)=>v.e[v.end+'Port']=(i+1)/(list.length+1));
 for(const e of g.edges){const [a,p]=port(lookup[e.from],e.sourceSide,e.sPort),[b,q]=port(lookup[e.to],e.targetSide,e.tPort);const middle=search(p,q,g.nodes,g.width,g.height,used);e.points=simplify([a,p,...middle,q,b]);e.via=[];if(e.label){e.labelBox=labelBox(e.points,e.label,g.nodes,labels,g.width);labels.push(e.labelBox);}}
}
export function roundedPath(points,radius=7){
 let d='M'+points[0].x+','+points[0].y;
 for(let i=1;i<points.length-1;i++){const a=points[i-1],b=points[i],c=points[i+1],ab=Math.abs(b.x-a.x)+Math.abs(b.y-a.y),bc=Math.abs(c.x-b.x)+Math.abs(c.y-b.y),r=Math.min(radius,ab/2,bc/2),p={x:b.x+Math.sign(a.x-b.x)*r,y:b.y+Math.sign(a.y-b.y)*r},q={x:b.x+Math.sign(c.x-b.x)*r,y:b.y+Math.sign(c.y-b.y)*r};d+=' L'+p.x+','+p.y+' Q'+b.x+','+b.y+' '+q.x+','+q.y;}
 return d+' L'+points.at(-1).x+','+points.at(-1).y;
}

"""Single edge import followed by device-only graph arithmetic."""
from __future__ import annotations
import math
import numpy as np
import torch


class DeviceGraph:
    def __init__(self,graph=None,device='cuda:0',*,n=None,edges=None):
        self.device=torch.device(device)
        if graph is not None:
            if graph.is_directed() or graph.is_multigraph():
                raise ValueError('Undirected simple graph required')
            self.n=len(graph)
            if set(graph)!=set(range(self.n)):raise ValueError('Consecutive node IDs required')
            # This is input serialization, not graph feature computation.
            imported=np.asarray(list(graph.edges()),dtype=np.int64).reshape(-1,2)
            raw=torch.as_tensor(imported.T.copy(),device=self.device,dtype=torch.int64)
        else:
            self.n=int(n);raw=edges.to(device=self.device,dtype=torch.int64)
        self.raw=raw
        if raw.shape[1]:
            reversed_edges=raw.flip(0)[:,raw[0]!=raw[1]]
            directed=torch.cat([raw,reversed_edges],1)
        else:directed=raw
        values=torch.ones(directed.shape[1],device=self.device,dtype=torch.float64)
        self.adjacency=torch.sparse_coo_tensor(directed,values,(self.n,self.n)).coalesce()
        ij=self.adjacency.indices();nonloop=ij[0]!=ij[1]
        self.clean=torch.sparse_coo_tensor(ij[:,nonloop],self.adjacency.values()[nonloop],
            (self.n,self.n)).coalesce()
        self.neighbor_count=torch.sparse.sum(self.adjacency,dim=1).to_dense()
        self.degree=self.neighbor_count.clone()
        self.degree.scatter_add_(0,raw[0,raw[0]==raw[1]],torch.ones_like(raw[0,raw[0]==raw[1]],dtype=torch.float64))

    @classmethod
    def union(cls,first,second):
        return cls(device=first.device,n=first.n+second.n,
            edges=torch.cat([first.raw,second.raw+first.n],1))

    def prepared(self,dummy=False):
        isolated=torch.nonzero(self.degree==0,as_tuple=True)[0]
        edges=torch.cat([self.raw,torch.stack([isolated,isolated])],1)
        n=self.n
        if dummy:
            edges=torch.cat([edges,torch.full((2,1),n,device=self.device,dtype=torch.long)],1)
            n+=1
        return DeviceGraph(device=self.device,n=n,edges=edges)

    def dense(self,dtype=torch.float64,size=None):
        size=self.n if size is None else size
        matrix=torch.zeros((size,size),device=self.device,dtype=dtype)
        ij=self.adjacency.indices()
        matrix[ij[0],ij[1]]=self.adjacency.values().to(dtype)
        return matrix

    def sparse(self,dtype=torch.float32,clean=False):
        return (self.clean if clean else self.adjacency).to(dtype)

    def normalized(self,dtype=torch.float32,add_identity=False):
        matrix=self.adjacency
        if add_identity:
            index=torch.arange(self.n,device=self.device)
            identity=torch.sparse_coo_tensor(torch.stack([index,index]),torch.ones(self.n,
                device=self.device,dtype=torch.float64),(self.n,self.n))
            matrix=(matrix+identity).coalesce()
        degree=torch.sparse.sum(matrix,dim=1).to_dense().clamp_min(1)
        ij=matrix.indices();weight=degree.rsqrt()
        return torch.sparse_coo_tensor(ij,(matrix.values()*weight[ij[0]]*weight[ij[1]]).to(dtype),
            matrix.shape).coalesce()

    def local_features(self,*,exclude_self=False,clustering=True,dtype=torch.float64,block=128):
        matrix=self.clean if exclude_self else self.adjacency
        count=torch.sparse.sum(matrix,dim=1).to_dense()
        coefficient=torch.zeros(self.n,device=self.device,dtype=torch.float64)
        if clustering:
            clean_degree=torch.sparse.sum(self.clean,dim=1).to_dense()
            for start in range(0,self.n,block):
                stop=min(self.n,start+block)
                identity=torch.zeros((self.n,stop-start),device=self.device,dtype=torch.float64)
                identity[torch.arange(start,stop,device=self.device),torch.arange(stop-start,device=self.device)]=1
                adjacent=torch.sparse.mm(self.clean,identity)
                second=torch.sparse.mm(self.clean,adjacent)
                denominator=clean_degree[start:stop]*(clean_degree[start:stop]-1)
                coefficient[start:stop]=torch.where(denominator>0,
                    (adjacent*second).sum(0)/denominator.clamp_min(1),0.)
        neighbor_degree=torch.sparse.mm(matrix,self.degree[:,None]).ravel()/count.clamp_min(1)
        neighbor_cluster=torch.sparse.mm(matrix,coefficient[:,None]).ravel()/count.clamp_min(1)
        return torch.stack([self.degree,coefficient,neighbor_degree,neighbor_cluster],1).to(dtype)

    def degree_features(self):
        ij=self.adjacency.indices()
        values=self.degree.float().log1p()
        mean=torch.zeros_like(values).scatter_add_(0,ij[0],values[ij[1]])/self.neighbor_count.float().clamp_min(1)
        variance=torch.zeros_like(values).scatter_add_(0,ij[0],
            (values[ij[1]]-mean[ij[0]]).square())/self.neighbor_count.float().clamp_min(1)
        return torch.stack([values,mean,variance.sqrt()],1)

    def hop_histograms(self,max_layer,alpha,buckets,block=128):
        maximum=self.n-1 if max_layer==0 else max_layer
        degree=self.degree.long()
        if buckets==1:
            # Degree is <=N+1 for a simple graph including self loops. A
            # fixed width avoids reading a device maximum on the host.
            width=self.n+2;indices=degree
            valid=torch.ones_like(degree,dtype=torch.bool)
        else:
            width=max(1,int(math.log(max(1,self.n+1),buckets))+1)
            indices=(degree.clamp_min(1).double().log2()/math.log2(buckets)).floor().long()
            valid=degree>0
        bucket=torch.nn.functional.one_hot(indices,width).double()*valid[:,None]
        output=torch.zeros((self.n,width),device=self.device,dtype=torch.float64)
        for start in range(0,self.n,block):
            stop=min(self.n,start+block)
            frontier=torch.zeros((self.n,stop-start),device=self.device,dtype=torch.float64)
            frontier[torch.arange(start,stop,device=self.device),torch.arange(stop-start,device=self.device)]=1
            visited=frontier.bool()
            for hop in range(min(maximum,self.n-1)+1):
                output[start:stop]+=(frontier.T@bucket)*alpha**hop
                reached=(torch.sparse.mm(self.clean,frontier)>0)&~visited
                visited|=reached;frontier=reached.double()
        return output


def topology_pair(first,second,adapter):
    if adapter=='constant_onehot3':
        bins=[torch.zeros(g.n,device=g.device,dtype=torch.long) for g in (first,second)]
    else:
        thresholds=torch.quantile(torch.cat([first.degree,second.degree]),
            torch.tensor([1/3,2/3],device=first.device,dtype=torch.float64))
        bins=[torch.bucketize(g.degree,thresholds,right=True) for g in (first,second)]
    return tuple(torch.nn.functional.one_hot(value,3).float() for value in bins)

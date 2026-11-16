"""Exact repeated-neighborhood/feature quotient of the full FUGAL iterates.

Full soft iterates are constant on these classes from uniform initialization.
Q_ab is TOTAL block mass, p_uv=Q_ab/(a_a*b_b); no node or candidate is removed.
OT entropy differs from full entropy only by fixed marginal-dependent terms.
"""
from dataclasses import dataclass
import numpy as np
import torch

from baselines.fugal import _original_features


@dataclass
class ExactClasses:
    membership: np.ndarray
    counts: np.ndarray
    representatives: np.ndarray
    features: np.ndarray
    adjacency: np.ndarray


def exact_classes(graph,padded_size):
    original=_original_features(graph,np.float64)
    features=np.zeros((padded_size,4),np.float64)
    features[:len(graph)]=original
    classes={};representatives=[];membership=[]
    for u in range(padded_size):
        signature=(tuple(sorted(graph[u])) if u<len(graph) else (),features[u].tobytes())
        if signature not in classes:
            classes[signature]=len(classes);representatives.append(u)
        membership.append(classes[signature])
    membership=np.asarray(membership,np.int64)
    representatives=np.asarray(representatives,np.int64)
    counts=np.bincount(membership,minlength=len(classes)).astype(np.float64)
    adjacency=np.zeros((len(classes),len(classes)),np.float64)
    for u,v in graph.edges():
        adjacency[membership[u],membership[v]]=1.
        adjacency[membership[v],membership[u]]=1.
    # Identical adjacency rows can only be false twins (no within-class edge).
    if np.any(np.diag(adjacency)):
        raise ValueError("A class is not an exact repeated adjacency row")
    expanded_edges=float(counts@adjacency@counts)/2
    if expanded_edges!=graph.number_of_edges():
        raise ValueError("Quotient failed exact edge-multiplicity preservation")
    return ExactClasses(membership,counts,representatives,features[representatives],adjacency)


def quotient_gradient(q,adjacency1,adjacency2,features,counts1,counts2,outer):
    p=q/counts1[:,None]/counts2[None,:]
    return -(adjacency1.T@q@adjacency2)-(adjacency1@q@adjacency2.T)+features+float(outer)*(1.-2.*p)


def expand_values(values,source,target,n1,n2):
    output=np.empty((n1,n2),np.float64)
    columns=target.membership[:n2]
    for begin in range(0,n1,128):
        output[begin:begin+128]=values[source.membership[begin:min(n1,begin+128)],:][:,columns]
    return output

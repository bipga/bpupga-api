"""Bundled native node classification metrics."""
import numpy as np

def node_classification(pairs, gt, n1, n2):
    """Matched and unmatched classes; GT-complement caveat is in protocol."""
    output = {}
    for label, predicted, truth, n in (("source", {u for u,v in pairs}, set(gt), n1),
                                      ("target", {v for u,v in pairs}, set(gt.values()), n2)):
        tp, fp, fn = len(predicted & truth), len(predicted-truth), len(truth-predicted)
        tn = n-tp-fp-fn
        div = lambda a,b:a/b if b else 0.0
        p,r = div(tn,tn+fn),div(tn,tn+fp)
        output[label] = {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"accuracy":div(tp+tn,n),
                         "balanced_accuracy":0.5*(div(tp,tp+fn)+div(tn,tn+fp)) if (tp+fn) and (tn+fp) else None,
                         "unmatched_precision":p,"unmatched_recall":r,"unmatched_f1":div(2*p*r,p+r),
                         "negative_class_present":bool(tn+fp),
                         "balanced_accuracy_defined_for_both_classes":bool(tp+fn) and bool(tn+fp)}
    return output

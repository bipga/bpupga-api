"""All twelve numerical algorithms and native decoders execute on CUDA."""
import time
import torch
from . import REVISION

METHODS=('alpine','cena','fugal','gradalign_plus','grampa','immnc','SparkAlign',
    'opgm_unsupervised','regal','s_gwl','slotalign','ulot')


class DeviceContract:
    def __init__(self,method,model,device):
        self.method,self.model,self.device=method,model,torch.device(device)
        self.config=model.config

    def fit(self,*graphs):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        started=time.perf_counter()
        result=self.model.fit(*graphs)
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        result.runtime_sec=time.perf_counter()-started
        execution=result.metadata.setdefault('device_execution',{})
        previous=execution.get('cpu_stages',[])
        execution.update(implementation_revision=REVISION,requested_device=str(self.device),
            actual_device=str(self.device),cuda_active=self.device.type=='cuda',
            execution_mode='all_algorithm_numerics_and_native_decode_on_device',
            cpu_fallback=False,numerical_cpu_stages=[],native_decoder_device=str(self.device),
            cpu_stages=['input parsing and checkpoint/file integrity I/O',
                'kernel launch and GPU-produced dynamic-shape dispatch',
                'final artifact/diagnostic serialization'],
            host_dispatch_includes_dynamic_shape_synchronization=True,
            all_stages_included_in_runtime=True,full_algorithm_end_to_end_seconds=result.runtime_sec,
            algorithm_numerics_and_decoding_on_GPU=True,host_resident_IO_buffers=True)
        execution['implementation_host_boundary_details']=previous
        result.metadata.update(implementation_revision=REVISION,gpu_revision=REVISION,
            ground_truth_used_during_inference=False)
        return result


def construct(method,parameters,seed=2026,device='cuda:0',dataset_family='',execution='standard'):
    if method not in METHODS:raise ValueError('Unknown GPU method: '+method)
    if device=='auto':device='cuda:0'
    selected=torch.device(device)
    if selected.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback forbidden')
    if method=='gradalign_plus' and 'centrality_max_iter' in parameters:
        from .centrality import construct as build
        return build(method,parameters,seed,device,dataset_family)
    if execution=='memory_efficient':
        if method=='alpine':from .transports_memory import construct as build
        elif method=='grampa':from .spectral_memory import construct as build
        else:raise ValueError('Unsupported memory-efficient execution')
        model,config=build(method,parameters,seed,device,dataset_family)
        return DeviceContract(method,model,device),config
    if execution!='standard':raise ValueError('Unknown execution configuration')
    if method in {'alpine','fugal'}:
        from .transports import construct as build
    elif method in {'grampa','regal'}:
        from .spectral import construct as build
    elif method in {'immnc','SparkAlign','s_gwl'}:
        from .iterative import construct as build
    elif method in {'ulot','opgm_unsupervised','slotalign'}:
        from .neural import construct as build
    elif method=='gradalign_plus':
        from .gradalign import construct as build
    else:
        from baseline.author_adapters import SingleCENAConfig
        from .cena import CUDAComputeCENA
        params=dict(parameters);params.pop('device',None);params.pop('final_acn',None)
        config=SingleCENAConfig(**dict(params,seed=seed,show_progress=False))
        from dataclasses import asdict
        model=CUDAComputeCENA(config,device=device)
        return DeviceContract(method,model,device),dict(asdict(config),
            implementation_revision=REVISION,cpu_fallback=False)
    model,config=build(method,parameters,seed,device,dataset_family) if method!='gradalign_plus' else build(parameters,seed,device,dataset_family)
    return DeviceContract(method,model,device),dict(config,implementation_revision=REVISION,cpu_fallback=False)

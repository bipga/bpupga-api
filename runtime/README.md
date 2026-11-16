# BiPGA

Twelve graph alignment methods with graph construction, native evaluation,
diagnostics, paper-result reproduction and plotting.

## Install

Use Python 3.10 for the complete twelve-method environment. Install the wheel from the [downloads page](https://bipga.github.io/installation.html), or install this source directory:

```bash
python -m pip install ".[all]"
bipga doctor --all
bipga methods
```

The complete runtime archive also contains `install.sh`, `bipga.sh`, a wheel and
the complete paper artifact. Run `bash install.sh --cpu` for CPU methods, or
`bash install.sh --gpu` for a CUDA environment. Installation does not fit models.

## Try the methods

```bash
bipga demo --method all --device cpu --output demo_cpu
bipga demo --method all --device cuda:0 --output demo_gpu
```

The installation example uses small generated graphs and reduced optimization
budgets. Use `--defaults` for adapter defaults. Paper parameters and recorded
inputs are supplied in the paper artifact.

## Align graph files

Put `G1.edgelist`, `G2.edgelist` and optional `G1.nodes`, `G2.nodes` in one folder.
`Alignment.json` supplies evaluation correspondences. Inference receives graph
topology and parameters in a separate process; reference labels stay in the
parent process for evaluation.

```bash
bipga run --data graph_pair --method grampa --parameters '{"eta":1.0}' \
    --device cpu --output results --diagnostics
bipga evaluate --data graph_pair --fit results/fit --output metrics.json
```

Existing output directories are never overwritten. Results include native
predictions, same-fit continuous outputs, effective configuration, metrics and
status records. The Python API is available as `import bpupga as bp` or
`import bipga as bp`:

```python
import bpupga as bp

data = bp.Dataset.load("graph_pair")
method = bp.Algorithm("grampa", parameters={"eta": 1.0}, device="cpu")
result = method.align(data.graph1, data.graph2)
print(bp.evaluate(result, data)["pair"])
```

## Construction and paper results

```bash
bipga construct --data raw_pair --rho-source 0.8 --rho-target 0.8 --output partial
bipga download --output paper
bipga reproduce --artifact paper --output reproduced
bipga plot --data-root reproduced/data --figure 5 --output figure05.pdf
bipga plot --data-root reproduced/data --figure 9 --output figure09
bipga paper --artifact paper -- --list --dataset fbtw
```

The paper artifact includes the 323 graph pairs, 3,908 configurations,
observations and the complete graph-to-figure reproduction guide. Replaying
observations uses a CPU and does not fit models. Full model fitting and the paper
CUDA assignment routine require an NVIDIA GPU and `nvcc`. Runtime and memory
depend on hardware. Follow the [supplementary guide](https://bipga.github.io/supplementary/reproduce/)
for exact paper execution and diagnostic protocols.

## GPU container

Build the supplied image definition:

```bash
docker build -f docker/Dockerfile -t bipga-gpu .
docker run --rm --gpus all bipga-gpu doctor --all --require-gpu
docker run --rm --gpus all -v "$PWD:/work" bipga-gpu \
    demo --method all --device cuda:0 --output /work/demo_gpu
```

The host needs an NVIDIA driver, Docker and NVIDIA Container Toolkit. The image
contains pinned PyTorch, CUDA development tools and method dependencies.

## Component terms

Bundled implementations retain their source notices and licenses. See
[component notices](THIRD_PARTY_NOTICES.md) and [license information](LICENSE).

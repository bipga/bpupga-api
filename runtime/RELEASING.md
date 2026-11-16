# Release administration

The Python wheel, source archive, runtime archive and GPU image use the same
package sources and dependencies. The public website links to the tested files.

## Build locally

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

The repository workflow is `.github/workflows/runtime-release.yml` in
`bipga/bpupga-api`. It checks an installed wheel on a fresh runner and builds the
GPU image from `runtime/docker/Dockerfile`. The normal push workflow does not
publish to PyPI.

## Enable PyPI trusted publishing

Use the owning PyPI account to add a pending publisher at
https://pypi.org/manage/account/publishing/:

| Field | Value |
| --- | --- |
| PyPI project | `bipga` |
| GitHub owner | `bipga` |
| Repository | `bpupga-api` |
| Workflow | `runtime-release.yml` |
| Environment | `pypi` |

For TestPyPI, create the equivalent pending publisher at
https://test.pypi.org/manage/account/publishing/ with environment `testpypi`.
PyPI and TestPyPI accounts and publishers are configured independently. Verify
name availability in the owning account before registration.

After configuring trust, open the workflow's **Run workflow** form and select
`testpypi` or `pypi` as the publishing target. Upload uses GitHub OIDC and requires
no token in the repository or downloaded bundle.

## GPU registry visibility

The workflow publishes `ghcr.io/bipga/bipga:0.1.0`. GitHub Container Registry can
create a package with private visibility. Set the package to **Public** at
https://github.com/users/bipga/packages/container/bipga/settings before advertising
anonymous pulls. Verify a pull without registry credentials. The source Docker
build remains available independently of registry visibility.

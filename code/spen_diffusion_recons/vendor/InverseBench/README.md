# InverseBench source subset

Only the SPEN operator base class and optional DAPS dependencies are copied:
`inverse_problems/base.py`, `algo/{base,daps}.py`, and `utils/{scheduler,diffusion}.py`.
Upstream code and the [LICENSE](LICENSE) are unchanged. This is a source subset,
not a complete standalone InverseBench checkout. The migration manifest records
resolved source paths and SHA256 hashes. No models, datasets or Git history were copied.

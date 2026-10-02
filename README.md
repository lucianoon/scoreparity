# scoreparity

**Prove that a change to your ML system did not change what your model outputs.**

Refactors, library upgrades (PyTorch, CUDA, scikit-learn), moving from CPU to GPU, quantization,
rewriting preprocessing, migrating from a notebook to a pipeline or between platforms: all of
these are supposed to *preserve* model behaviour. In practice they are usually checked by eye,
often by comparing an aggregate metric such as AUC. That is not enough: AUC can stay identical
while individual scores move enough to flip decisions, and a "no significant difference"
t-test passes precisely when the evidence is weakest.

`scoreparity` compares the scores of a **reference** and a **candidate** version on the same
records and answers a single question with sound statistics: *are they equivalent within the
tolerance you declared up front?* It is model-agnostic. All it needs is two tables of scores.

> Status: early development (0.1.0.dev). The API and CLI are not stable yet.

## License

Apache-2.0. See [LICENSE](LICENSE).

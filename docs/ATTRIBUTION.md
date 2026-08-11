# Attribution

This repository is an independent educational implementation. It was informed by the
following references and dependencies:

- Sebastian Raschka, *Build a Large Language Model (From Scratch)* and the accompanying
  [official code repository](https://github.com/rasbt/LLMs-from-scratch). The implementation
  here is written independently; book code and visuals are not copied into this repository.
- OpenAI, [GPT-2](https://github.com/openai/gpt-2), for the original model architecture and
  published reference implementation.
- The [Hugging Face `openai-community/gpt2` checkpoint](https://huggingface.co/openai-community/gpt2),
  used only for opt-in compatibility verification.
- [PyTorch](https://pytorch.org/), [tiktoken](https://github.com/openai/tiktoken), and
  [Transformers](https://github.com/huggingface/transformers), used as documented optional
  dependencies.
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), used through
  streaming data access. The dataset is published under ODC-By; any future dataset-derived
  artifacts must preserve its required attribution.

The project code is released under the MIT License in the repository root. Third-party
projects and datasets retain their own licenses.

## OpenAI GPT-2 notice

The upstream GPT-2 repository publishes its code under a modified MIT License and asks users
to preserve its copyright and responsible-use notice when redistributing that upstream code.
This project does not redistribute OpenAI source code or model weights; it implements the
architecture independently and downloads the reference checkpoint only for opt-in compatibility
verification. If upstream assets are added in the future, their original license notice must be
included separately. See the [upstream license](https://github.com/openai/gpt-2/blob/master/LICENSE).

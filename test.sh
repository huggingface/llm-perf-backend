docker run -it --rm ghcr.io/huggingface/optimum-benchmark:latest-cpu


docker run -it --rm --pid host --volume "$(pwd)":/optimum-benchmark --workdir /optimum-benchmark ghcr.io/huggingface/optimum-benchmark:latest-cpu
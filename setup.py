from setuptools import setup, find_packages

# Base dependencies
INSTALL_REQUIRES = [
    "typer",
    "python-dotenv",
    "ruff",
    "packaging",
    "einops",
    "scipy",
    "optimum",
    "codecarbon",
    "transformers",
    "huggingface_hub[hf_transfer]",
    "datasets>=2.14.6",
    "beautifulsoup4",

    "optimum-benchmark @ git+https://github.com/huggingface/optimum-benchmark.git",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "onnxruntime": [
        "onnx",
        "onnxruntime",
        "optimum-benchmark[onnxruntime] @ git+https://github.com/huggingface/optimum-benchmark.git",
    ],
    "openvino": [
        "optimum-benchmark[openvino] @ git+https://github.com/huggingface/optimum-benchmark.git"
    ],
    "cuda": [
        "flash-attn",
        "auto-gptq",
        "bitsandbytes",
        "autoawq",
    ],
    "dashboard": [
        "gradio>=5.0.0",
        "sentence-transformers",
    ]
}

setup(
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "llm-perf = llm_perf.cli:app",  # Referring to 'cli.py' inside 'src' directory
        ]
    },
    name="llm-perf-backend",
    version="0.1.0",
    description="Backend for https://huggingface.co/spaces/optimum/llm-perf-leaderboard",
    author="baptiste",
    author_email="baptiste.colle@huggingface.co",
    packages=find_packages(),  # This ensures that Python looks for packages inside 'src'
)

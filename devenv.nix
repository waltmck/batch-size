{ pkgs, lib, config, ... }:

let
  # The NVIDIA *driver* libraries live in Ubuntu's lib dir, which the Nix dynamic
  # loader never searches. Expose just those few libraries (not all of /usr/lib)
  # to the Python interpreter so the pip-installed JAX CUDA plugin can dlopen libcuda.
  nvidiaDriverLibs = pkgs.runCommand "nvidia-driver-libs" { } ''
    mkdir -p $out/lib
    for f in libcuda.so.1 libcudadebugger.so.1 libnvidia-ml.so.1 libnvidia-nvvm.so.4 libnvidia-ptxjitcompiler.so.1; do
      ln -s /usr/lib/x86_64-linux-gnu/$f $out/lib/$f
    done
  '';
in
{
  # Python 3.12 from nixpkgs; the ML stack comes from PyPI (manylinux wheels) via a venv in .devenv/state/venv.
  # Versions pinned to what the authors used in their Colab notebooks (jax 0.7.2 + flax 0.12.0).
  languages.python = {
    enable = true;
    package = pkgs.python312;
    # Libraries prefixed onto LD_LIBRARY_PATH by the Python wrapper (devenv adds libstdc++/libgcc itself).
    # Deliberately NOT `manylinux.enable = true`: that package also puts Nix's libc on the path, which breaks
    # the Ubuntu-built `ptxas` binary XLA spawns from the nvidia-cuda-nvcc wheel (GLIBC_PRIVATE symbol clash).
    libraries = [ nvidiaDriverLibs pkgs.zlib ];
    venv.enable = true;
    venv.requirements = ''
      jax[cuda12]==0.7.2
      flax==0.12.0
      optax
      chex          # imported by pretraining/factorized.py (no longer pulled in by optax)
      numpy
      tqdm
      hydra-core
      wandb
      fire
      huggingface_hub

      # finetuning/ additionally needs: datasets kagglehub sentencepiece math_verify, plus qwix from GitHub (LoRA)

      # notebooks / analysis
      jupyterlab
      ipykernel
      papermill
      matplotlib
      pandas
      seaborn
    '';
  };

  env = {
    # Don't let one JAX process grab 75% of the GPU up front, so a notebook kernel and a
    # background run can share the A100 (the 30M model needs only a few GB).
    XLA_PYTHON_CLIENT_PREALLOCATE = "false";
    # Keep tqdm from flooding logs / notebook outputs on million-step runs.
    TQDM_MININTERVAL = "30";
  };

  enterShell = ''
    echo "batch-size devenv: $(python --version)"
  '';
}

{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  packages = with pkgs; [
    git
    python313Packages.cupy
    python313Packages.numpy
    python313Packages.pandas
    python313Packages.flask
    python313Packages.flask-cors
    python313Packages.gunicorn
    python313Packages.keras
    python313Packages.tqdm
  ];
  tasks = {
    "omp:setcores" = {
      exec = ''
        export OPENBLAS_CORETYPE=skylake
        export OMP_NUM_THREADS=8
      '';
      before = ["devenv:enterShell"];
    };
  };
  languages.python.enable = true;
}

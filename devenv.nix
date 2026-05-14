{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  packages = with pkgs; [
    git
    python313Packages.numpy
    python313Packages.pandas
    python313Packages.flask
    python313Packages.flask-cors
    python313Packages.gunicorn
  ];

  languages.python.enable = true;

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };
}

{
  description = "Tensor-based Spline Utilities";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.05";

    utils.url = "github:numtide/flake-utils";
    utils.inputs.nixpkgs.follows = "nixpkgs";

    ml-pkgs.url = "github:nixvital/ml-pkgs";
    ml-pkgs.inputs.nixpkgs.follows = "nixpkgs";
    ml-pkgs.inputs.utils.follows = "utils";
  };

  outputs = { self, nixpkgs, utils, ... }@inputs: utils.lib.eachSystem [
    "x86_64-linux"
  ] (system: let
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
      overlays = [(final: prev: rec {
        python3 = prev.python3.override {
          packageOverrides = pyFinal: pyPrev : rec {
            inherit (inputs.ml-pkgs.packages."${system}")
              pytorchWithCuda11;
          };
        };
        python3Packages = python3.pkgs;
      })];
    };
  in {
    devShells = {
      default = pkgs.mkShell rec {
        name = "tensor-splines";

        packages = let pythonEnv = pkgs.python3.withPackages (pyPkgs: with pyPkgs; [
          numpy
          matplotlib
          pytorchWithCuda11
          jupyterlab
          ipywidgets
        ]); in [
          pythonEnv
          # Dev Tools
          pkgs.nodePackages.pyright
          pkgs.pre-commit
        ];

        shellHook = ''
          export PS1="$(echo -e '\uf3e2') {\[$(tput sgr0)\]\[\033[38;5;228m\]\w\[$(tput sgr0)\]\[\033[38;5;15m\]} (${name}) \\$ \[$(tput sgr0)\]"
          export PYTHONPATH="$(pwd):$PYTHONPATH"
        '';
      };

      poetry = pkgs.mkShell rec {
        name = "tensor-splines";
        packages = with pkgs; [
          poetry
        ];
        shellHook = ''
          export PS1="$(echo -e '\uf3e2') {\[$(tput sgr0)\]\[\033[38;5;228m\]\w\[$(tput sgr0)\]\[\033[38;5;15m\]} (${name}) \\$ \[$(tput sgr0)\]"
          export PYTHONPATH="$(pwd):$PYTHONPATH"
          # In poetry environment, packages such as numpy needs them.
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib/:${pkgs.zlib}/lib/:$LD_LIBRARY_PATH"
      '';
      };
    };
  });
}

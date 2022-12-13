{
  description = "Tensor-based Spline Utilities";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";

    utils.url = "github:numtide/flake-utils";

    ml-pkgs.url = "github:nixvital/ml-pkgs";
    ml-pkgs.inputs.nixpkgs.follows = "nixpkgs";
    ml-pkgs.inputs.utils.follows = "utils";
  };

  outputs = { self, nixpkgs, utils, ... }@inputs: {
    overlays.default = final: prev: {
      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (python-final: python-prev: {
          tensor-splines = python-final.callPackage ./nix/pkgs/tensor-spline {
            pytorch = python-final.pytorchWithCuda11;
          };
        })
      ];
    };
  } // utils.lib.eachSystem [
    "x86_64-linux"
  ] (system: let
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
      overlays = [
        inputs.ml-pkgs.overlays.torch-family
      ];
    };
  in {
    packages.default = pkgs.python3Packages.callPackage ./nix/pkgs/tensor-spline {};

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

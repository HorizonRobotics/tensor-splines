{ lib
, buildPythonPackage
, numpy
, matplotlib
, pytorch
, poetry-core
}:

buildPythonPackage rec {
  pname = "tensor-splines";
  version = "0.5.0";
  format = "pyproject";

  src = ../../..;

  nativeBuildInputs = [
    poetry-core
  ];

  propagatedBuildInputs = [
    numpy
    matplotlib
    pytorch
  ];

  meta = with lib; {
    homepage = "https://github.com/HorizonRobotics/tensor-splines";
    description = ''
      An utility library to generate splines with tensor-based batch operation support
    '';
    license = licenses.mit;
    maintainers = with maintainers; [ breakds ];
  };
}

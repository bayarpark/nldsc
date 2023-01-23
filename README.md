## NLDSC - Additive & Non-Additive LD Score Regression




## Prerequisites
- `Python>=3.8`
- `CMake>=3.20`, `Ninja`
- `Armadillo`, `gfotran`, `BLAS`/`OpenBLAS`, `LAPACK`

## Installation
**Ubuntu 20.04, 22.04:**
Firstly, install essential libraries: 
```
sudo apt update
sudo apt install -y ninja-build gfortran libblas-dev liblapack-dev libarmadillo-dev
```

Make sure you have installed all libraries from the prerequisites. Then, run the following commands:
```
git clone --recursive https://github.com/bayarpark/nldsc.git
cd nldsc
pip install -r requirements.txt
make build
```

**Install via Conda** (Linux and MacOS only) 

`%todo%`


## LD Score Estimation

```bash
nldsc ld --bfile <prefix/to/[.bed|.bim|.fam]> --ld-wind-cm 1 -maf 0.0001 --std-thr 1e-5 --out <prefix/to/result>
```

## Heritability Estimation
```bash
%otod%
```

---

### LICENSE
GNU GPL. See [License File](https://github.com/bayarpark/nldsc/blob/master/LICENSE)

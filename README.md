## `nldsc` — (non)-additive LD Score Regression

This project is an extension of the LD Score Regression ([`ldsc`](https://github.com/bulik/ldsc)), a widely-used method for estimating the heritability of complex traits using summary statistics from genome-wide association studies (GWAS). This extension allows to take into account non-additive effects (*but we consider only dominance for now*), which have been shown to play a significant role in the genetic architecture of many complex traits. 


### Implementation Details

- The `ldscore` module has been written from scratch, but we took into account key ideas from the original ldsc package.
- The `h2` module heavily builds on the original implementation and has mostly been refactored.
- The console interface is completely different from `ldsc`, with only the names of some flags being saved.


### Precomputed LD Scores
You can find an `.tar.gz` archive with additive and non-additive LD Scores pre-calculated on UK Biobank data ($N=315599$) in the "Releases" tab.


## Getting Started
### Prerequisites
- `Python>=3.8`
- `CMake>=3.20`, `Ninja`
- `Armadillo`, `gfotran`, `BLAS`/`OpenBLAS`, `LAPACK`

**Ubuntu 20.04, 22.04:**
```bash
sudo apt update
sudo apt install -y ninja-build gfortran libblas-dev liblapack-dev libarmadillo-dev
```


### Installation

#### Install manually
Please ensure that you have installed all required packages. Then, clone the repository:

```bash
git clone --recursive https://github.com/bayarpark/nldsc.git
cd nldsc
pip install -r requirements.txt
make build
```

#### Install via Conda (Linux and MacOS only) 
```
%under development%
```


## Usage
#### LD Score Estimation

```bash
python nldsc ld --bfile <prefix/to/[.bed|.bim|.fam]> --ld-wind-cm 1 -maf 0.0001 --std-thr 1e-5 --out <prefix/to/result>
```

#### Heritability Estimation
```bash
python nldsc h2 --sumstats <path/to/ldsc-like.sumstats> --ref-ld <path/to/ld> --w-ld <path/to/ld> 
```


## LICENSE
This project is licensed under the GNU GPL-3.0. See [License File](https://github.com/bayarpark/nldsc/blob/master/LICENSE)

## Authors
Bayar Park (Novosibirsk State University)

This package (especially the `h2` module) is largely based on the sources of the [`ldsc`](https://github.com/bulik/ldsc) package written by Brendan Bulik-Sullivan (Broad Institute of MIT and Harvard) and Hilary Finucane (MIT Department of Mathematics)




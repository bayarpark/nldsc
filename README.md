## NLDSC - Additive & Non-Additive LD Score Regression




## Prerequisites
**Ubuntu 20.04:**
1) Install Python 3.8 or higher
2) Install CMake version 3.20 or higher [directly](https://cmake.org/install/) or using [this](https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line) instruction. 
4) Then run the following:
```
sudo apt update
sudo apt install -y ninja-build gfortran libblas-dev liblapack-dev libarmadillo-dev
```


## Installation
Make sure you have installed all libraries from the prerequisites.
```
git clone --recursive https://github.com/bayarpark/nldsc.git
cd nldsc
pip install -r requirements.txt
make build
```


## Usage sample

**Calculation of additive and non-additive LD Scores**
```
./nldsc.py -ld -p <path/to/[.bed|.bim|.fam]> --ld-wind-cm 1 --maf 0.01 --out <path/to/result.csv>
```

---

### LICENSE
GNU GPL. See [License File](https://github.com/bayarpark/nldsc/blob/master/LICENSE)

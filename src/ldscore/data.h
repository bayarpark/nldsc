#pragma once

#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#define ARMA_NO_DEBUG


#include <string>
#include <utility>
#include <map>
#include <iostream>
#include <string>
#include <ios>
#include <fstream>
#include <vector>
#include <armadillo>
#include <cmath>


typedef unsigned char uchar;

const unsigned char SIGNIFICANT_TWO_BITS = 192;

namespace BedBitEnc {
    const uchar HOM_FST = 0;    //  00	Homozygous for first allele in .bim file
    const uchar MISS    = 64;   //  01	Missing genotype
    const uchar HET     = 128;  //  10	Heterozygous
    const uchar HOM_SND = 192;  //  11	Homozygous for second allele in .bim file
}



/* >>>
 *      Own structures
 */


struct PairSNP {
public:
    arma::fvec* add_ = nullptr;
    arma::fvec* nadd_ = nullptr;
    bool use_ = false;
    float maf_ = 0;

    PairSNP() = default;
    PairSNP(arma::fvec* add, arma::fvec* nadd, float maf) {
        add_ = add;
        nadd_ = nadd;
        maf_ = maf;
        use_ = true;
    }

    void reset() {
        if (use_) {
            add_->reset();
            nadd_->reset();
            delete add_;
            delete nadd_;
            use_ = false;
        }
    }
};


struct LDScoreParams {
    std::string bedfile;                // Absolute path to .BED file
    int num_of_snp;                     // Number of SNPs
    int num_of_org;                     // Number of organism
    double ld_wind;                     // Size of LD Windows
    double maf;
    std::vector<double> positions;

    LDScoreParams() = default;
};


#ifndef LDSCORE_DATA_H
#define LDSCORE_DATA_H

#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#define ARMA_NO_DEBUG

#include <armadillo>

#include <string>
#include <utility>
#include <map>
#include <iostream>
#include <string>
#include <ios>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>


typedef unsigned char uchar;

const unsigned char SIGNIFICANT_TWO_BITS = 192;
const float MISSING = arma::datum::nan;


namespace BEDBinaryEncoding {
    const uchar HOM_FST = 0;    //  00	Homozygous for first allele in .bim file
    const uchar MISS    = 64;   //  01	Missing genotype
    const uchar HET     = 128;  //  10	Heterozygous
    const uchar HOM_SND = 192;  //  11	Homozygous for second allele in .bim file
}


struct LDScoreParams {
    std::string bedfile;                // Absolute path to .BED file
    int num_of_snp;                     // Number of SNPs
    int num_of_org;                     // Number of organism
    double ld_wind;                     // Size of LD Windows
    double maf;                         // TODO
    std::vector<double> positions;      // TODO Vector with

    LDScoreParams() = default;
};

#endif //LDSCORE_DATA_H

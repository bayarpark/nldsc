#pragma once

#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#define ARMA_NO_DEBUG

#include <armadillo>

#include <string>
#include <iostream>
#include <string>
#include <ios>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>


using uchar = unsigned char;

struct LDScoreResult {
    std::vector<double> l2;             // Additive LD Scores
    std::vector<double> l2d;            // Non-additive LD Scores

    std::vector<double> maf;            // Minor allele frequencies
    std::vector<double> residuals_std;  // Standard deviation for residuals

    std::vector<int> l2_ws;             // (Effective) Number of SNPs in the window (additive)
    std::vector<int> l2d_ws;            // Number of SNPs in the window (non-additive)
    std::vector<int> l2d_wse;           // Effective number of SNPs in the window (non-additive)
};

struct LDScoreParams {
    std::string bedfile;                // Absolute path to .BED file

    int n_snp;                          // Number of SNPs
    int n_org;                          // Number of organism

    double ld_wind;                     // Size of LD Windows (cM or kbp)
    std::vector<double> positions;      // Coordinates in genome (cM or kbp)

    double maf;                         // Minor allele frequency threshold
    double std_thr;                     // Variance threshold for non-additive residuals
    double rsq_thr;                     // Threshold for non-additive r-squared coefficients


    LDScoreParams() = default;
    LDScoreParams(const std::string& bedfile,
                  int n_snp,
                  int n_org,
                  double ld_wind,
                  double maf,
                  double std_threshold,
                  double rsq_threshold,
                  const std::vector<double>& positions) {

        this->bedfile = bedfile;
        this->n_snp = n_snp;
        this->n_org = n_org;
        this->ld_wind = ld_wind;
        this->maf = maf;
        this->std_thr = std_threshold;
        this->rsq_thr = rsq_threshold;
        this->positions = positions;
    }
};

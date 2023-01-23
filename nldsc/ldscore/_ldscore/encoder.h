#pragma once

#include "data.h"
#include "tools.h"

#include <memory>


const float MISSING = arma::datum::nan;

namespace BEDBinaryEncoding {
    const uchar HOM_FST = 0;    //  00	Homozygous for first allele in .bim file
    const uchar MISS    = 64;   //  01	Missing genotype
    const uchar HET     = 128;  //  10	Heterozygous
    const uchar HOM_SND = 192;  //  11	Homozygous for second allele in .bim file
}

template<int HOM_FST, int HET, int HOM_SND>
inline float encoding(uchar val) {
    switch (val) {
        case BEDBinaryEncoding::HOM_FST:
            return HOM_FST;
        case BEDBinaryEncoding::MISS:
            return MISSING;
        case BEDBinaryEncoding::HET:
            return HET;
        case BEDBinaryEncoding::HOM_SND:
            return HOM_SND;
        default:
            throw std::runtime_error("Invalid binary encoding");
    }
}

inline float additive(uchar val) {
    return encoding<0, 1, 2>(val);
}

inline float dominant(uchar val) {
    return encoding<0, 2, 2>(val);
}


class SNPInMemory {
private:
    std::unique_ptr<arma::fvec> add_;           // Standardised SNP in additive encoding
    std::unique_ptr<arma::fvec> residuals_;     // Standardised non-additive residuals
    float maf_ = MISSING;                       // SNP minor allele frequency
    float residuals_std_ = MISSING;             // Standard deviation of residuals
    bool use_ = false;                          // Use this SNP or not

public:
    SNPInMemory() = default;

    SNPInMemory(const std::vector<uchar>& raw, double maf_thr) {
        use_ = decode(raw, maf_thr);
        if (use_) {
            Math::standardise(*add_);
            residuals_std_ = Math::standardise(*residuals_);
        }
    }

    [[nodiscard]] double maf() const {
        return maf_;
    }

    [[nodiscard]] double residuals_std() const {
        return residuals_std_;
    }

    [[nodiscard]] arma::fvec add() const {
        return *add_;
    }

    [[nodiscard]] arma::fvec residuals() const {
        return *residuals_;
    }

    explicit operator bool() const {
        return use_;
    }

    void release() {
        if (use_) {
            add_.reset();
            residuals_.reset();
            use_ = false;
        }
    }

private:
    bool decode(const std::vector<uchar>& raw, double maf_thr) {
        add_ = std::make_unique<arma::fvec>(raw.size());
        arma::fvec nadd(raw.size());

        double add_sum = 0;
        double nadd_sum = 0;
        int n_nans = 0;

        for (uint i = 0; i < raw.size(); ++i) {
            auto val = raw[i];
            auto add_val = additive(val);
            auto nadd_val = dominant(val);

            if (val != BEDBinaryEncoding::MISS) {
                add_sum += add_val;
                nadd_sum += nadd_val;
                ++n_nans;
            }

            add_->at(i) = add_val;
            nadd.at(i) = nadd_val;
        }

        auto add_mean = static_cast<float>(add_sum / n_nans);
        auto nadd_mean = static_cast<float>(nadd_sum / n_nans);

        float f2 = add_mean / 2;
        maf_ = f2 < 0.5 ? f2 : 1 - f2;

        if (maf_ <= maf_thr) {
            add_.reset();
            return false;
        } else {
            for (uint i = 0; i < raw.size(); ++i) {
                if (std::isnan(add_->at(i))) { [[unlikely]]
                                                       add_->at(i) = add_mean;
                    nadd.at(i) = nadd_mean;
                }
            }
            residuals_ = std::make_unique<arma::fvec>(Math::regression_residuals(*add_, nadd));
            return true;
        }
    }
};

class EmptySNP : public SNPInMemory{
private:
    EmptySNP() = default;;
public:
    static EmptySNP& instance() {
        static EmptySNP instance;
        return instance;
    }
    EmptySNP(EmptySNP const &) = delete;
    void operator=(EmptySNP const &) = delete;
};

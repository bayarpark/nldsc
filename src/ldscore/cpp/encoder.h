#ifndef LDSCORE_ENCODER_H
#define LDSCORE_ENCODER_H

#include "data.h"
#include "tools.h"



template<int HOM_FST, int HET, int HOM_SND>
inline float
encoding(uchar val) {
    if (val == BEDBinaryEncoding::HOM_FST) {
        return HOM_FST;
    } else if (val == BEDBinaryEncoding::MISS) {
        return MISSING;
    } else if (val == BEDBinaryEncoding::HET) {
        return HET;
    } else if (val == BEDBinaryEncoding::HOM_SND) {
        return HOM_SND;
    } else {
        auto msg = "Invalid binary encoding";
        throw std::logic_error(msg);
    }
}


inline float
additive(uchar val) {
    return encoding<0, 1, 2>(val);
}

inline float
dominant(uchar val) {
    return encoding<0, 2, 2>(val);
}


struct SNPInMemory {
private:
    arma::fvec add_;         // Standardised SNP in additive encoding
    arma::fvec residuals_;   // Standardised non-additive residuals
    float maf_ = MISSING;    //
    float residuals_std_;    //
    bool use_ = false;       //


public:

    SNPInMemory() = default;
    SNPInMemory(float* mem_add, float* mem_nadd, float maf, uint n_elem) {

        this->add_ = arma::fvec(mem_add, n_elem);
        this->residuals_ = Math::regression_residuals(add_, arma::fvec(mem_nadd, n_elem));
        this->maf_ = maf;
        this->use_ = true;
        Math::standardise(this->add_);

        Math::standardise(this->residuals_, &(this->residuals_std_));

        delete[] mem_add;
        delete[] mem_nadd;
    }

    double
    maf() const {
        return maf_;
    }

    double
    residuals_std() const {
        return residuals_std_;
    }

    const arma::fvec&
    add() const {
        return add_;
    }

    const arma::fvec&
    residuals() const {
        return residuals_;
    }

    void release() {
        if (use_) {
            add_.reset();
            residuals_.reset();
            use_ = false;
        }
    }
    explicit operator bool() const { return use_; }
};


class EmptySNP : public SNPInMemory{
private:
    EmptySNP() {};
public:
    static EmptySNP& instance() {
        static EmptySNP instance;
        return instance;
    }
    EmptySNP(EmptySNP const &) = delete;
    void operator=(EmptySNP const &) = delete;
};


inline SNPInMemory
apply_encoding(const uchar *vec, uint n_elem, double maf_threshold) {
    auto *add = new float[n_elem];
    auto *nadd = new float[n_elem];

    double add_sum = 0;
    double nadd_sum = 0;
    int nnans = 0;

    for (uint i = 0; i < n_elem; ++i) {
        auto val = vec[i];

        auto add_val = additive(val);
        auto nadd_val = dominant(val);

        if (val != BEDBinaryEncoding::MISS) {
            add_sum += add_val;
            nadd_sum += nadd_val;
            ++nnans;
        }

        add[i] = add_val;
        nadd[i] = nadd_val;
    }

    auto add_mean = static_cast<float>(add_sum / nnans);
    auto nadd_mean = static_cast<float>(nadd_sum / nnans);

    float f2 = add_mean / 2;
    float maf = f2 < 0.5 ? f2 : 1 - f2;

    if (maf <= maf_threshold) {
        return EmptySNP::instance();
    }

    for (uint i = 0; i < n_elem; ++i) {
        if (std::isnan(add[i])) {
            add[i] = add_mean;
            nadd[i] = nadd_mean;
        }
    }
    return SNPInMemory{add, nadd, maf, n_elem};
}

#endif //LDSCORE_ENCODER_H

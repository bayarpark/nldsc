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
    arma::fvec add_;
    arma::fvec nadd_;
    arma::fvec residuals_;

    bool ev_add_norm_ = false;
    bool ev_nadd_norm_ = false;
    bool ev_residuals_norm_ = false;

public:
    float maf_ = 0;
    bool use_ = false;

    SNPInMemory() = default;
    SNPInMemory(float *add, float *nadd, float maf, uint n_elem) {
        this->add_ =  arma::fvec(add, n_elem);
        this->nadd_ = arma::fvec(nadd, n_elem);
        delete[] add;
        delete[] nadd;
        this->maf_ = maf;
        this->use_ = true;
    }


    const arma::fvec&
    add() {
        if (not ev_add_norm_) {
            Math::standardise(add_);
            ev_add_norm_ = true;
        }
        return add_;
    }

    const arma::fvec&
    nadd() {
        if (not ev_nadd_norm_) {
            Math::standardise(nadd_);
            ev_nadd_norm_ = true;
        }
        return nadd_;
    }


    const arma::fvec&
    residuals() {
        if (not ev_residuals_norm_) {
            residuals_ = Math::regression_residuals(this->add(), this->nadd());
            Math::standardise(residuals_);
            ev_residuals_norm_ = true;
        }
        return residuals_;
    }

    void release() {
        if (use_) {
            add_.reset();
            nadd_.reset();
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
apply_encoding(const uchar *vec, uint n_elem) {
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

    for (uint i = 0; i < n_elem; ++i) {
        if (std::isnan(add[i])) {
            add[i] = add_mean;
            nadd[i] = nadd_mean;
        }
    }
    return SNPInMemory{add, nadd, maf, n_elem};
}

#endif //LDSCORE_ENCODER_H

#pragma once
#include "data.h"



inline float additive(uchar val)
{
    if (val == BedBitEnc::HOM_FST) {
        return 0;
    } else if (val == BedBitEnc::MISS) {
        return arma::datum::nan;
    } else if (val == BedBitEnc::HET) {
        return 1;
    } else if (val == BedBitEnc::HOM_SND) {
        return 2;
    } else {
        auto msg = "Invalid binary encoding";
        throw std::logic_error(msg);
    }
}

inline float dominant(uchar val) {
    if (val == BedBitEnc::HOM_FST) {
        return 0;
    } else if (val == BedBitEnc::MISS) {
        return arma::datum::nan;
    } else if (val == BedBitEnc::HET or val == BedBitEnc::HOM_SND) {
        return 2;
    } else {
        auto msg = "Invalid binary encoding";
        throw std::logic_error(msg);
    }
}

inline PairSNP apply_encoding(const arma::uchar_vec& vec) {
    auto* add = new arma::fvec(vec.n_elem);
    auto* nadd = new arma::fvec(vec.n_elem);

    double add_sum = 0;
    double nadd_sum = 0;
    int nnans = 0;

    for (arma::uword i = 0; i < vec.n_elem; ++ i) {
        auto val = vec(i);

        auto add_val = additive(val);
        auto nadd_val = dominant(val);

        if (val != BedBitEnc::MISS) {
            add_sum += add_val;
            nadd_sum += nadd_val;
            ++nnans;
        }

        (*add)(i) = add_val;
        (*nadd)(i) = nadd_val;
    }

    auto add_mean = static_cast<float>(add_sum / nnans);
    auto nadd_mean = static_cast<float>(nadd_sum / nnans);

    float f2 = add_mean / 2;
    float maf = f2 < 0.5 ? f2 : 1 - f2;

    for (arma::uword i = 0; i < vec.n_elem; ++ i) {
        if (std::isnan((*add)(i))) {
            (*add)(i) = add_mean;
            (*nadd)(i) = nadd_mean;
        }
    }

    return PairSNP{add, nadd, maf};
}


class Encoder {
private:
    const float MISSING = arma::datum::nan;
    std::map<uchar, float> NADD;

    const std::map<uchar, float> ADD{
            {BedBitEnc::HOM_FST, 0},
            {BedBitEnc::MISS,    MISSING},
            {BedBitEnc::HET,     1},
            {BedBitEnc::HOM_SND, 2}
    };

    const std::map<uchar, float> DOM{
            {BedBitEnc::HOM_FST, 0},
            {BedBitEnc::MISS,    MISSING},
            {BedBitEnc::HET,     2},
            {BedBitEnc::HOM_SND, 2}
    };


public:
    Encoder() {
        this->NADD = DOM;
    }

    Encoder& operator=(const Encoder& x) {
        return *this;
    }

    float nonadditive(uchar val) const{
        return NADD.at(val);
    }

    float additive(uchar val) const {
        return ADD.at(val);
    }

    arma::fvec nonadditive(const arma::uchar_vec& vec) const{
        arma::fvec out = arma::fvec(vec.n_elem);
        double sum = 0;
        int n_nans = 0;
        for (arma::uword i = 0; i < vec.n_elem; ++ i) {
            auto val = this->nonadditive(vec(i));
            out(i) = val;
            if (vec(i) != BedBitEnc::MISS) {
                sum += val;
                ++n_nans;
            }
        }
        out = out.replace(MISSING, sum / n_nans);
        return out;
    }

    arma::fvec additive(const arma::uchar_vec& vec) const{
        arma::fvec out = arma::fvec(vec.n_elem);
        double sum = 0;
        int n_nans = 0;
        for (arma::uword i = 0; i < vec.n_elem; ++ i) {
            auto val = this->additive(vec(i));
            out(i)  = val;
            if (vec(i) != BedBitEnc::MISS) {
                ++n_nans;
                sum += val;
            }
        }
        out = out.replace(MISSING, sum / n_nans);
        return out;
    }

    PairSNP apply(const arma::uchar_vec& vec) const {
        arma::fvec* add = new arma::fvec(vec.n_elem);
        arma::fvec* nadd = new arma::fvec(vec.n_elem);

        double add_sum = 0;
        double nadd_sum = 0;
        int nnans = 0;


        for (arma::uword i = 0; i < vec.n_elem; ++ i) {
            auto val = vec(i);

            auto add_val = additive(val);
            auto nadd_val = nonadditive(val);

            if (val != BedBitEnc::MISS) {
                add_sum += add_val;
                nadd_sum += nadd_val;
                ++nnans;
            }

            (*add)(i) = add_val;
            (*nadd)(i) = nadd_val;
        }

        auto add_mean = static_cast<float>(add_sum / nnans);
        auto nadd_mean = static_cast<float>(nadd_sum / nnans);

        float f2 = add_mean / 2;
        float maf = f2 < 0.5 ? f2 : 1 - f2;

        for (arma::uword i = 0; i < vec.n_elem; ++ i) {
            if (std::isnan((*add)(i))) {
                (*add)(i) = add_mean;
            }
            if (std::isnan((*nadd)(i))) {
                (*nadd)(i) = nadd_mean;
            }
        }

        return PairSNP{add, nadd, maf};
    }
};


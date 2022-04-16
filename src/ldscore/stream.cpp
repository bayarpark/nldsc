#pragma once

#include "tools.cpp"
#include "encoder.cpp"
#include "data.h"


class BedStreamReader {
    /**
     *
     */
public:
    BedStreamReader() = default;

    explicit BedStreamReader(LDScoreParams &params) {
        this->params_ = params;
        this->stream_ = std::ifstream(params.bedfile, std::ios::binary);
        this->n_blocks_ = params_.num_of_org / 4 + (params_.num_of_org % 4 > 0);
        this->buf = new char[n_blocks_];
        this->check_plink_magic_number();
    }

    inline arma::uchar_vec read() {
        /**
         * Reads and returns the next SNP from a file
         *
         * @return Armadillo vector with specific encoding (see Encoder)
         */
        ++curr_snp_;
        if (curr_snp_ > params_.num_of_snp) {
            return {};
        }
        arma::uchar_vec snp = arma::uchar_vec(params_.num_of_org);
        int bitpairs = 4;

        stream_.read(buf, n_blocks_);
        for (int j = 0; j < n_blocks_; ++j) {
            uchar curr_byte = buf[j];
            if ((j == n_blocks_ - 1) and (params_.num_of_org % 4 != 0)) {
                bitpairs = params_.num_of_org % 4;
            }
            for (int i = 0; i < bitpairs; ++i) {
                snp[4*j + i] = curr_byte & SIGNIFICANT_TWO_BITS;
                curr_byte = (curr_byte ^ (curr_byte & SIGNIFICANT_TWO_BITS)) << 2;
            }
        }
        return snp;
    }

    inline void pass() {
        /**
         * Passes the next SNP.
         */
        ++curr_snp_;
        if (curr_snp_ < params_.num_of_snp) {
            stream_.ignore(n_blocks_);
        }
    };

private:
    LDScoreParams params_;
    std::ifstream stream_;
    int curr_snp_ = 0;
    int n_blocks_ = 0;
    char* buf = nullptr;

    void check_plink_magic_number() {
        /**
         * Checks if the magic number of the .BED file is correct.
         *
         * @throw
         *
         */
        char fst, snd, trd;
        stream_.get(fst);
        stream_.get(snd);
        stream_.get(trd);

        if (not ((fst == 0x6c) and (snd == 0x1b) and (trd == 0x01))) { // FAIL
            auto msg = "Invalid PLINK magic number in BED file."
                       "The file is incorrect, or it was created using an old version of PLINK.";
            throw std::invalid_argument(msg);
        }
    }
};


class ChunkCache {
public:

    ChunkCache() = default;

    explicit ChunkCache(LDScoreParams &params) {
        this->params_ = params;
        this->reader_ = BedStreamReader(params);
        this->filter_ = SNPFilter(params);
        this->curr_chunk_.reserve(8192);
        this->cache_.reserve(params_.num_of_snp);
    }

    inline std::pair<arma::fvec*, std::vector<PairSNP>&> next_chunk() {
        ++curr_snp_;
        extend_cache();
        curr_chunk_.resize(0);

        for (int i = left_snp_; i <= right_snp_; ++i) {
            if (filter_.filter(curr_snp_, i) and cache_[i].use_) {
                if (i != curr_snp_)
                    curr_chunk_.push_back(cache_[i]);
            } else if (left_snp_ == i and left_snp_ < curr_snp_) {
                cache_[left_snp_].reset();
                ++left_snp_;
            }
        }
        return {cache_[curr_snp_].add_, curr_chunk_};
    }

    void pass_chunk() {
        ++curr_snp_;
    }


private:
    LDScoreParams params_;
    BedStreamReader reader_;
    SNPFilter filter_;
    std::vector<PairSNP> curr_chunk_;
    std::vector<PairSNP> cache_;

    int left_snp_ = 0;
    int curr_snp_ = -1;
    int right_snp_ = -1;

    inline void extend_cache() {
        do {
            if (right_snp_ + 1 >= params_.num_of_snp)
                break;
            ++right_snp_;

            if (filter_.is_used(right_snp_)) {
                auto snp = reader_.read();
                auto encoded = apply_encoding(snp); // encoder_.apply(snp);
                if (filter_.filter_maf(encoded.maf_)) {
                    cache_.push_back(encoded);
                } else {
                    cache_.emplace_back();
                }
            } else {
                cache_.emplace_back();
                reader_.pass();
            }
        } while (filter_.filter(curr_snp_, right_snp_));
    }
};

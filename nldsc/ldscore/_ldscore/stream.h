#pragma once

#include <memory>

#include "tools.h"
#include "encoder.h"
#include "data.h"

class BedStreamReader {
    /**
     * BED File reader (SNP-wise; raw)
     */
public:
    BedStreamReader() = default;

    explicit BedStreamReader(LDScoreParams& params)
            : params_(params)
            , stream_(std::ifstream(params.bedfile, std::ios::binary))
            , buf(new char[2048*2048]) {

        n_blocks_ = params_.n_org / 4 + (params_.n_org % 4 > 0);
        stream_.rdbuf()->pubsetbuf(buf, 2048*2048);
        check_plink_magic_number();
    }

    BedStreamReader(const BedStreamReader&) = delete;

    BedStreamReader(BedStreamReader&& rhs) noexcept
            : params_(std::move(rhs.params_))
    , stream_(std::move(rhs.stream_))
    , curr_snp_(rhs.curr_snp_)
    , n_blocks_(rhs.n_blocks_)
    , buf(new char[2048*2048]) {
        stream_.rdbuf()->pubsetbuf(buf, 2048*2048);
    };



    ~BedStreamReader() {
        delete[] buf;
    };

    std::vector<uchar> read() {
        /**
         * Reads the next SNP from the .BED file
         *
         * @return array (uchar*) in raw encoding (see Encoder)
         */
        ++curr_snp_;
        if (curr_snp_ > params_.n_snp)
            return {};

        std::vector<uchar> snp(params_.n_org);

        int bitpairs = 4;
        char byte;
        for (int j = 0; j < n_blocks_; ++j) {
            stream_.get(byte);

            if ((j == n_blocks_ - 1) and (params_.n_org % 4 != 0))
                bitpairs = params_.n_org % 4;

            for (int i = 0; i < bitpairs; ++i) {
                snp[4*j + i] = byte & 192;
                byte = (byte ^ (byte & 192)) << 2;
            }
        }
        return snp;
    }

    void pass() {
        /**
         * Passes the next SNP
         */
        ++curr_snp_;
        if (curr_snp_ < params_.n_snp) {
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
         */
        char fst, snd, trd;
        stream_.get(fst);
        stream_.get(snd);
        stream_.get(trd);

        if (not ((fst == 0x6c) and (snd == 0x1b) and (trd == 0x01))) { // FAIL
            auto msg = "Invalid PLINK magic number in BED file."
                       "The file is incorrect, or it was created using an incompatible version of PLINK.";
            throw std::invalid_argument(msg);
        }
    }
};


class ChunkwiseReader {
    /**
    * Cached sliding window through BED file
    */
private:
    LDScoreParams params_;
    BedStreamReader reader_;
    SNPFilter filter_;
    std::vector<SNPInMemory> cache_;

    int left_snp_ = 0;
    int curr_bottom_ = left_snp_;
    int curr_snp_ = -1;
    int right_snp_ = -1;

public:
    ChunkwiseReader() = default;

    explicit ChunkwiseReader(LDScoreParams &params)
            : params_(params)
            , reader_(BedStreamReader(params))
            , filter_(SNPFilter(params)) {
        this->cache_.reserve(params_.n_snp);
    }

    bool initialize_next_chunk() {
        ++curr_snp_;
        extend_cache();
        curr_bottom_ = left_snp_;
        return curr_snp_ < params_.n_snp and cache_[curr_snp_];
    }

    arma::fvec y() const {
        return cache_[curr_snp_].add();
    }

    inline std::vector<uint> chunk_indices() {
        std::vector<uint> indices;
        for (int i = left_snp_; i <= right_snp_; ++i) {
            if (cache_[i] and filter_.filter(curr_snp_, i)) {
                if (i != curr_snp_) {
                    indices.push_back(i);
                }
            } else if (left_snp_ == i and left_snp_ < curr_snp_) {
                cache_[left_snp_].release();
                ++left_snp_;
            }
        }
        return indices;
    }

    void pass_chunk() {
        ++curr_snp_;
    }

    const SNPInMemory& operator[](uint idx) const {
        return cache_[idx];
    }

    std::vector<double> mafs() const {
        std::vector<double> rv(params_.n_snp);
        for (int i = 0; i < params_.n_snp; ++i) {
            rv[i] = cache_[i].maf();
        }
        return rv;
    }

    std::vector<double> residual_stds() const {
        std::vector<double> rv(params_.n_snp);
        for (int i = 0; i < params_.n_snp; ++i) {
            rv[i] = cache_[i].residuals_std();
        }
        return rv;
    }

private:
    void extend_cache() {
        do {
            if (right_snp_ + 1 >= params_.n_snp)
                break;

            ++right_snp_;

            if (filter_.is_used(right_snp_)) {
                const auto& snp = reader_.read();
                cache_.emplace_back(snp, params_.maf);
            } else {
                cache_.emplace_back();
                reader_.pass();
            }
        } while (filter_.filter(curr_snp_, right_snp_));
    }
};

#ifndef LDSCORE_STREAM_H
#define LDSCORE_STREAM_H


#include "tools.h"
#include "encoder.h"
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
        this->buf = new char[1024*1024];
        this->stream_.rdbuf()->pubsetbuf(buf, 1024*1024);
        this->check_plink_magic_number();
    }

    inline uchar* read() {
        /**
         * Reads and returns the next SNP from the .BED file
         *
         * @return array (uchar*) in specific encoding (see Encoder)
         */
        ++curr_snp_;
        if (curr_snp_ > params_.num_of_snp) {
            return {};
        }
        auto* snp = new uchar [params_.num_of_org];
        int bitpairs = 4;
        char curr_byte;

        //stream_.read(buf, n_blocks_);
        for (int j = 0; j < n_blocks_; ++j) {
            stream_.get(curr_byte);
            //uchar curr_byte = buf[j];
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
                       "The file is incorrect, or it was created using an incompatible version of PLINK.";
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
        this->cache_.reserve(params_.num_of_snp);
        fout.open("read_time.log");
    }

    bool init_next_chunk() {
        ++curr_snp_;
        extend_cache();
        curr_bottom_ = left_snp_;
        return curr_snp_ < params_.num_of_snp and cache_[curr_snp_];
    }


    const arma::fvec&
    get_add() const {
        return cache_[curr_snp_].add();
    }

    void pass_chunk() {
        ++curr_snp_;
    }

    const SNPInMemory& get() {
        for (int i = curr_bottom_; i <= right_snp_; ++i) {
            if (cache_[i] and filter_.filter(curr_snp_, i)) {
                if (i != curr_snp_) {
                    curr_bottom_ = i + 1;
                    return cache_[i];
                }
            } else if (left_snp_ == i and left_snp_ < curr_snp_) {
                cache_[left_snp_].reset();
                ++left_snp_;
            }
        }
        return bottom_;
    }


private:
    LDScoreParams params_;
    BedStreamReader reader_;
    SNPFilter filter_;
    std::vector<SNPInMemory> cache_;

    SNPInMemory bottom_;

    int left_snp_ = 0;
    int curr_bottom_ = left_snp_;
    int curr_snp_ = -1;
    int right_snp_ = -1;


    std::ofstream fout;


    inline void extend_cache() {
        do {
            if (right_snp_ + 1 >= params_.num_of_snp)
                break;
            ++right_snp_;

            if (filter_.is_used(right_snp_)) {
                auto snp = reader_.read();
                auto encoded = apply_encoding(snp, this->params_.num_of_org);
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

#endif // LDSCORE_STREAM_H

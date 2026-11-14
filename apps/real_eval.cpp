// real_eval.cpp -- the real Mimir protocol on a real database with real queries.
//
// Same client/server pipeline as benchmarks/benchmark_latency_seeded_parallel*.cpp
// (seeded eval keys and query, bit-packed wire streams, RLWE + RGSW switching,
// scoring over EVERY (cluster, split), RGSW masking, cross-cluster summation,
// client unpack + decrypt), with two differences:
//   - database polynomials are built from a real MDB1 file (mmap'ed int8,
//     embedding-major) instead of random values; they're converted to eval form
//     on the fly per query because 138M x 256 eval-form coefficients don't fit
//     in RAM (this adds server time vs. a resident-DB server; this run is for
//     correctness/quality, costs come from the benchmarks),
//   - the query is a real quantized embedding + its nearest-centroid cluster.
// Every decrypted score of the selected cluster is written out.
//
// Usage: ./real_eval <params.json> <db.mdb> <queries.bin> <out_dir> [first] [count]

#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "params_io.hpp"
#include "seeded_distribution.hpp"
#include "seeded_eval_keys.hpp"
#include "seeded_query.hpp"
#include "stream_packing.hpp"

using namespace FHEDeck;
using namespace psearch;
using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

struct Mdb {
    uint32_t C = 0, l = 0;
    std::vector<uint32_t> sizes;
    std::vector<const int8_t*> cluster;
};

static Mdb map_mdb(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("open " + path);
    struct stat st;
    fstat(fd, &st);
    auto* base = static_cast<const uint8_t*>(mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0));
    if (base == MAP_FAILED) throw std::runtime_error("mmap " + path);
    const uint32_t* h = reinterpret_cast<const uint32_t*>(base);
    if (h[0] != 0x4D444231) throw std::runtime_error("bad magic");
    Mdb m;
    m.C = h[1];
    m.l = h[2];
    m.sizes.assign(h + 3, h + 3 + m.C);
    size_t off = 12 + 4 * static_cast<size_t>(m.C);
    for (uint32_t c = 0; c < m.C; ++c) {
        m.cluster.push_back(reinterpret_cast<const int8_t*>(base + off));
        off += static_cast<size_t>(m.sizes[c]) * m.l;
    }
    if (off != static_cast<size_t>(st.st_size)) throw std::runtime_error("size mismatch");
    return m;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <params.json> <db.mdb> <queries.bin> <out_dir> [first] [count]\n";
        return 1;
    }
    Params params = load_params_from_json(argv[1], 1, 0);
    CryptoContext ctx = CryptoContext::from_params(params);
    if (params.num_component_rings != 1) throw std::runtime_error("only r=1 parameter sets supported");
    const std::string out_dir = argv[4];
    const int64_t n = params.n, l = params.embedding_length, S = params.splits_per_cluster, C = params.num_clusters;

    Mdb db = map_mdb(argv[2]);
    if (db.C != C || db.l != l) throw std::runtime_error("db shape != params");
    for (uint32_t c = 0; c < db.C; ++c)
        if (db.sizes[c] != params.cluster_size) throw std::runtime_error("non-uniform cluster sizes");
    const int64_t lo = min_embedding_value(params), hi = max_embedding_value(params);
    std::cerr << "db: C=" << C << " cluster_size=" << params.cluster_size << " splits=" << S
              << " threads=" << omp_get_max_threads() << "\n";

    std::ifstream qf(argv[3], std::ios::binary);
    uint32_t Q = 0;
    qf.read(reinterpret_cast<char*>(&Q), 4);
    std::vector<int32_t> qcluster(Q);
    std::vector<std::vector<int8_t>> qvals(Q, std::vector<int8_t>(static_cast<size_t>(l)));
    for (uint32_t i = 0; i < Q; ++i) {
        qf.read(reinterpret_cast<char*>(&qcluster[i]), 4);
        qf.read(reinterpret_cast<char*>(qvals[i].data()), l);
    }
    int64_t first = argc > 5 ? std::stoll(argv[5]) : 0;
    int64_t count = argc > 6 ? std::stoll(argv[6]) : static_cast<int64_t>(Q) - first;

    // --- Registration (client keygen -> packed stream -> server reconstructs from the stream only).
    ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
    SeededClientPublicMaterial eval_wire = build_seeded_public_material(ctx, secret);
    ByteStream key_stream = pack_public_material(params, eval_wire);
    ClientPublicMaterial pub = reconstruct_public_material(ctx, params, unpack_public_material(params, key_stream));
    std::cerr << "registered, eval key stream " << key_stream.size() << " B\n";

    std::ofstream log(out_dir + "/log.tsv", std::ios::app);
    log << "query\tcluster\tclient_query_s\tserver_s\tclient_decrypt_s\tupload_B\tdownload_B\n";

    for (int64_t qi = first; qi < first + count && qi < static_cast<int64_t>(Q); ++qi) {
        std::vector<int64_t> emb(static_cast<size_t>(l));
        for (int64_t j = 0; j < l; ++j) {
            int64_t v = qvals[qi][static_cast<size_t>(j)];
            if (v < lo || v > hi) throw std::runtime_error("query value out of range");
            emb[static_cast<size_t>(j)] = reduce_mod(v, params.plaintext_modulus);
        }
        auto t0 = Clock::now();
        SeededQuery query_wire = build_seeded_query(ctx, params, secret, emb, qcluster[qi]);
        std::vector<ByteStream> up{pack_query_ring(params, query_wire, 0)};
        auto t1 = Clock::now();

        // ------------------------------- server -------------------------------
        ReconstructedQuery query = reconstruct_query(ctx, params, unpack_query(params, up));
        std::vector<std::unique_ptr<RLWECTEvalForm>> qe(static_cast<size_t>(l));
        #pragma omp parallel for schedule(dynamic)
        for (int64_t j = 0; j < l; ++j) {
            RLWECT ct(ctx.rlwe_param);
            pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(ct, query.embedding_cts[0][static_cast<size_t>(j)]);
            qe[static_cast<size_t>(j)] = std::make_unique<RLWECTEvalForm>(ct);
        }
        std::vector<std::unique_ptr<RLWEGadgetCT>> rgsw(query.selector_cts.size());
        #pragma omp parallel for schedule(dynamic)
        for (int64_t c = 0; c < static_cast<int64_t>(query.selector_cts.size()); ++c) {
            rgsw[static_cast<size_t>(c)] = std::make_unique<RLWEGadgetCT>(
                pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(query.selector_cts[static_cast<size_t>(c)]));
        }
        std::vector<std::vector<std::unique_ptr<RLWECT>>> masked(static_cast<size_t>(S));
        for (auto& row : masked) row.resize(static_cast<size_t>(C));
        #pragma omp parallel for schedule(dynamic)
        for (int64_t idx = 0; idx < C * S; ++idx) {
            int64_t c = idx / S, s = idx % S;
            const int8_t* data = db.cluster[static_cast<size_t>(c)];
            int64_t size = db.sizes[static_cast<size_t>(c)];
            RLWECTEvalForm acc(ctx.rlwe_param);
            std::vector<int64_t> raw(static_cast<size_t>(n));
            for (int64_t j = 0; j < l; ++j) {
                for (int64_t i = 0; i < n; ++i) {
                    int64_t e = s * n + i;
                    raw[static_cast<size_t>(i)] = e < size ? static_cast<int64_t>(data[e * l + j]) : 0;
                }
                DatabasePolynomialEvalForm p =
                    build_database_polynomial_eval_form_from_raw_values(ctx, params, raw, params.plaintext_modulus);
                RLWECTEvalForm prod(ctx.rlwe_param);
                qe[static_cast<size_t>(j)]->mul(prod, *p.poly_eval);
                acc.add(acc, prod);
            }
            RLWECT score(acc);
            RLWECT mv(ctx.rlwe_param);
            rgsw[static_cast<size_t>(c)]->mul(mv, score);
            masked[static_cast<size_t>(s)][static_cast<size_t>(c)] = std::make_unique<RLWECT>(mv);
        }
        std::vector<RLWECT> result;
        for (int64_t s = 0; s < S; ++s) {
            result.emplace_back(ctx.rlwe_param);
            for (int64_t c = 0; c < C; ++c)
                result.back().add(result.back(), *masked[static_cast<size_t>(s)][static_cast<size_t>(c)]);
        }
        ByteStream down = pack_response_ring(params, result);
        auto t2 = Clock::now();

        // ------------------------------- client -------------------------------
        std::vector<RLWECT> recv = unpack_response_ring(ctx, params, down);
        std::vector<int16_t> scores(static_cast<size_t>(S * n));
        #pragma omp parallel for schedule(dynamic)
        for (int64_t s = 0; s < S; ++s) {
            Vector dec = secret.rlwe_sk->decrypt_vector(recv[static_cast<size_t>(s)], ctx.encoding);
            std::vector<int64_t> v = decode_to_signed(dec, params);
            for (int64_t i = 0; i < n; ++i) scores[static_cast<size_t>(s * n + i)] = static_cast<int16_t>(v[static_cast<size_t>(i)]);
        }
        auto t3 = Clock::now();

        std::ofstream so(out_dir + "/q" + std::to_string(qi) + ".scores.i16", std::ios::binary);
        so.write(reinterpret_cast<const char*>(scores.data()), static_cast<std::streamsize>(scores.size() * 2));
        log << qi << "\t" << qcluster[qi] << "\t" << secs(t0, t1) << "\t" << secs(t1, t2) << "\t" << secs(t2, t3)
            << "\t" << up[0].size() << "\t" << down.size() << "\n" << std::flush;
        std::cerr << "q" << qi << " cluster " << qcluster[qi] << ": server " << secs(t1, t2) << " s\n";
    }
    return 0;
}

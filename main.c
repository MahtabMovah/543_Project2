// main.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <inttypes.h>
#include <stdbool.h>

#include "btree.h"
#include "hctree.h"

// Simple payload: just store the key as a pointer-sized value.
static void* make_payload(int64_t k) {
    return (void*)(intptr_t)k;
}

// Uniform random in [0, n-1]
static int64_t rand_uniform(int64_t n) {
    return (int64_t)((double)rand() / ((double)RAND_MAX + 1.0) * (double)n);
}

// Zipf sampler with precomputed CDF O(log N) sampling.
typedef struct {
    double  *cdf; // size N
    int64_t  N;
    double   s;
} ZipfGen;

static ZipfGen* zipf_create(int64_t N, double s) {
    ZipfGen *z = (ZipfGen*)malloc(sizeof(ZipfGen));
    z->N = N;
    z->s = s;
    z->cdf = (double*)malloc(sizeof(double) * N);

    double sum = 0.0;
    for (int64_t k = 1; k <= N; k++) {
        sum += 1.0 / pow((double)k, s);
    }

    double cumsum = 0.0;
    for (int64_t k = 1; k <= N; k++) {
        cumsum += 1.0 / pow((double)k, s) / sum;
        z->cdf[k-1] = cumsum;
    }
    return z;
}

static void zipf_free(ZipfGen *z) {
    if (!z) return;
    free(z->cdf);
    free(z);
}

static int64_t zipf_sample(ZipfGen *z) {
    double u = (double)rand() / ((double)RAND_MAX + 1.0);
    // binary search in cdf
    int64_t lo = 0, hi = z->N - 1, mid;
    while (lo < hi) {
        mid = (lo + hi) / 2;
        if (u <= z->cdf[mid]) hi = mid;
        else lo = mid + 1;
    }
    // lo is index in [0, N-1], representing rank (1-based)
    return lo; // treat as key in [0, N-1]
}

// For timing
static double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "Options:\n"
        "  --nkeys N         number of distinct keys (default 100000)\n"
        "  --nqueries Q      number of point queries (default 500000)\n"
        "  --workload TYPE   'uniform' or 'zipf' (default zipf)\n"
        "  --theta S         zipf exponent (default 1.1)\n"
        "  --hot_thresh H    hot threshold (default 8.0)\n"
        "  --decay A         decay alpha (default 0.9)\n"
        "  --hot_frac F      max hot fraction (default 0.05)\n"
        "  --seed SEED       RNG seed (default 42)\n"
        "  --mode MODE       'hctree' (default) or 'baseline'\n"
        "  --disable_hot     alias for --mode baseline\n"
        "  --sample_init D    initial sampling rate D in (0,1] (default 1.0)\n"
        "  --adapt_sample     enable adaptive (ML-style) tuning of D\n"
        "  --csv             output one line of CSV instead of human-readable text\n"
        "  --csv_header      print CSV header and exit\n",
        prog);
}

typedef enum {
    MODE_HCTREE = 0,
    MODE_BASELINE = 1
} RunMode;

int main(int argc, char **argv) {
    int64_t nkeys = 100000;
    int64_t nqueries = 500000;
    const char *workload = "zipf";
    double theta = 1.1;
    double hot_thresh = 8.0;
    double decay_alpha = 0.9;
    double hot_frac = 0.05;
    unsigned int seed = 42;
    RunMode mode = MODE_HCTREE;
    bool csv = false;
    bool csv_header = false;
    double sample_init   = 1.0;
    int    adapt_sample  = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--nkeys") && i+1 < argc) {
            nkeys = atoll(argv[++i]);
        } else if (!strcmp(argv[i], "--nqueries") && i+1 < argc) {
            nqueries = atoll(argv[++i]);
        } else if (!strcmp(argv[i], "--workload") && i+1 < argc) {
            workload = argv[++i];
        } else if (!strcmp(argv[i], "--theta") && i+1 < argc) {
            theta = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--hot_thresh") && i+1 < argc) {
            hot_thresh = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--decay") && i+1 < argc) {
            decay_alpha = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--hot_frac") && i+1 < argc) {
            hot_frac = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--seed") && i+1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--mode") && i+1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "hctree")) mode = MODE_HCTREE;
            else if (!strcmp(m, "baseline")) mode = MODE_BASELINE;
            else {
                fprintf(stderr, "Unknown mode '%s'\n", m);
                usage(argv[0]);
                return 1;
            }

        // 🔽 NEW OPTIONS START HERE
        } else if (!strcmp(argv[i], "--sample_init") && i+1 < argc) {
            // Initial sampling rate D (0–1)
            sample_init = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--adapt_sample")) {
            // Turn on ML-style adaptation of D
            adapt_sample = 1;
        // 🔼 NEW OPTIONS END HERE

        } else if (!strcmp(argv[i], "--disable_hot")) {
            mode = MODE_BASELINE;
        } else if (!strcmp(argv[i], "--csv")) {
            csv = true;
        } else if (!strcmp(argv[i], "--csv_header")) {
            csv_header = true;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (csv_header) {
        // Print header and exit; no experiment.
        printf("mode,workload,theta,nkeys,nqueries,hot_threshold,decay_alpha,hot_fraction,seed,"
               "elapsed_sec,qps,hot_hits,cold_hits,not_found,hot_keys,cold_keys,"
               "avg_hot_nodes_per_q,avg_cold_nodes_per_q\n");
        return 0;
    }

    srand(seed);

    int btree_degree = 32; // B-tree min degree (t)

    double t0, t1, elapsed, qps;

    long hot_hits = 0;
    long cold_hits = 0;
    long not_found = 0;
    size_t hot_keys = 0;
    size_t cold_keys = 0;
    double avg_hot_nodes_q = 0.0;
    double avg_cold_nodes_q = 0.0;

    ZipfGen *zg = NULL;
    if (!strcmp(workload, "zipf")) {
        zg = zipf_create(nkeys, theta);
    }

    if (mode == MODE_HCTREE) {
        // --- Hot/Cold index mode ---
        HCParams params;
        params.decay_alpha      = decay_alpha;
        params.hot_threshold    = hot_thresh;
        params.max_hot_fraction = hot_frac;
        params.inclusive        = 1;
        params.sampling_rate    = sample_init;   // NEW
        params.adapt_sampling   = adapt_sample;  // NEW

        if (!csv) {
            printf("Mode:       HCIndex (hot/cold)\n");
            printf("Workload:   %s\n", workload);
            if (!strcmp(workload, "zipf"))
                printf("Theta:      %.3f\n", theta);
            printf("nkeys:      %" PRId64 "\n", nkeys);
            printf("nqueries:   %" PRId64 "\n", nqueries);
            printf("HotThresh:  %.3f\n", hot_thresh);
            printf("Decay alpha:%.3f\n", decay_alpha);
            printf("Hot frac:   %.3f\n", hot_frac);
        }

        HCIndex *idx = hc_create(nkeys - 1, btree_degree, params);

        // Build cold index
        for (int64_t k = 0; k < nkeys; k++) {
            hc_insert(idx, k, make_payload(k));
        }

        t0 = now_seconds();
        for (int64_t q = 0; q < nqueries; q++) {
            int64_t k;
            if (!strcmp(workload, "zipf")) {
                k = zipf_sample(zg);
            } else {
                k = rand_uniform(nkeys);
            }
            (void)hc_search(idx, k);
        }
        t1 = now_seconds();

        HCStats s = hc_get_stats(idx);
        elapsed = t1 - t0;
        qps = (elapsed > 0.0) ? (double)nqueries / elapsed : 0.0;

        hot_hits = s.hot_hits;
        cold_hits = s.cold_hits;
        not_found = s.not_found;
        hot_keys = s.hot_keys;
        cold_keys = s.cold_keys;
        avg_hot_nodes_q  = s.queries ? (double)s.hot_node_visits  / (double)s.queries : 0.0;
        avg_cold_nodes_q = s.queries ? (double)s.cold_node_visits / (double)s.queries : 0.0;

        if (!csv) {
            printf("\n=== Results (HCIndex) ===\n");
            printf("Elapsed (sec):    %.6f\n", elapsed);
            printf("Throughput (Q/s): %.2f\n", qps);
            printf("Hot hits:         %ld\n", hot_hits);
            printf("Cold hits:        %ld\n", cold_hits);
            printf("Not found:        %ld\n", not_found);
            printf("Hot keys:         %zu\n", hot_keys);
            printf("Cold keys:        %zu\n", cold_keys);
            printf("Avg hot nodes/q:  %.3f\n", avg_hot_nodes_q);
            printf("Avg cold nodes/q: %.3f\n", avg_cold_nodes_q);

        }

        hc_free(idx);
    } else {
        // --- Baseline mode: single B-tree only ---
        if (!csv) {
            printf("Mode:       Baseline (single B-tree)\n");
            printf("Workload:   %s\n", workload);
            if (!strcmp(workload, "zipf"))
                printf("Theta:      %.3f\n", theta);
            printf("nkeys:      %" PRId64 "\n", nkeys);
            printf("nqueries:   %" PRId64 "\n", nqueries);
        }

        BTree *bt = bt_create(btree_degree);

        // Build baseline index
        for (int64_t k = 0; k < nkeys; k++) {
            bt_insert(bt, k, make_payload(k));
        }

        long total_node_visits = 0;
        long nf = 0;

        t0 = now_seconds();
        for (int64_t q = 0; q < nqueries; q++) {
            int64_t k;
            if (!strcmp(workload, "zipf")) {
                k = zipf_sample(zg);
            } else {
                k = rand_uniform(nkeys);
            }
            BTStats s = {0};
            void *v = bt_search(bt, k, &s);
            total_node_visits += s.node_visits;
            if (v == NULL)
                nf++;
        }
        t1 = now_seconds();

        elapsed = t1 - t0;
        qps = (elapsed > 0.0) ? (double)nqueries / elapsed : 0.0;

        not_found = nf;
        cold_hits = nqueries - nf;  // everything goes to "cold" conceptually
        hot_hits = 0;
        hot_keys = 0;
        cold_keys = bt_count_keys(bt);
        avg_hot_nodes_q = 0.0;
        avg_cold_nodes_q = nqueries ? (double)total_node_visits / (double)nqueries : 0.0;

        if (!csv) {
            printf("\n=== Results (Baseline) ===\n");
            printf("Elapsed (sec):    %.6f\n", elapsed);
            printf("Throughput (Q/s): %.2f\n", qps);
            printf("Cold hits:        %ld\n", cold_hits);
            printf("Not found:        %ld\n", not_found);
            printf("Cold keys:        %zu\n", cold_keys);
            printf("Avg nodes/q:      %.3f\n", avg_cold_nodes_q);
        }

        bt_free(bt);
    }

    if (zg) zipf_free(zg);

    if (csv) {
        // Single CSV line. Note: we still print hot_* fields for baseline (they'll be 0).
        const char *mode_str = (mode == MODE_HCTREE) ? "hctree" : "baseline";
        printf("%s,%s,%.5f,%" PRId64 ",%" PRId64 ",%.5f,%.5f,%.5f,%u,"
               "%.6f,%.2f,%ld,%ld,%ld,%zu,%zu,%.6f,%.6f\n",
               mode_str,
               workload,
               theta,
               nkeys,
               nqueries,
               hot_thresh,
               decay_alpha,
               hot_frac,
               seed,
               elapsed,
               qps,
               hot_hits,
               cold_hits,
               not_found,
               hot_keys,
               cold_keys,
               avg_hot_nodes_q,
               avg_cold_nodes_q);
    }

    return 0;
}

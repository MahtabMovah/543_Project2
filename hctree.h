// hctree.h
#ifndef HCTREE_H
#define HCTREE_H

#include "btree.h"

// Parameters controlling hot/cold behavior.
typedef struct {
    double decay_alpha;      // e.g., 0.9
    double hot_threshold;    // e.g., 8.0
    double max_hot_fraction; // e.g., 0.10 (10% of keys)
    int    inclusive;        // 1 = hot is a cache (no deletes in cold)

    // Sampling + ML-style adaptation knobs
    double sampling_rate;    // D in the paper, 0 < D <= 1
    int    adapt_sampling;   // 0 = fixed D, 1 = adaptive (ML) controller
} HCParams;

// Statistics for evaluation.
typedef struct {
    long queries;
    long hot_hits;
    long cold_hits;
    long not_found;

    long hot_node_visits;
    long cold_node_visits;

    size_t hot_keys;
    size_t cold_keys;
} HCStats;

// Main hot/cold index structure.
typedef struct {
    BTree  *hot;
    BTree  *cold;

    int64_t max_key;     // keys ∈ [0, max_key]
    double *hit_score;   // array[max_key+1]

    HCParams params;
    HCStats  stats;

    // --- Online linear regression state (classic ML) ---
    // We model: cost(D) ≈ w0 + w1 * D
    long   last_q_for_adapt;   // queries at last adaptation point
    long   last_hot_nodes;     // hot_node_visits at last adaptation
    long   last_cold_nodes;    // cold_node_visits at last adaptation
    double lr_w0;              // regression weight for bias
    double lr_w1;              // regression weight for D
} HCIndex;

HCIndex* hc_create(int64_t max_key, int btree_degree, HCParams params);
void     hc_free(HCIndex *idx);

// Build index: insert into COLD only (hot starts empty).
void     hc_insert(HCIndex *idx, BTKey k, BTPayload v);

// Point lookup: hot first, then cold if miss.
BTPayload hc_search(HCIndex *idx, BTKey k);

// Range search: returns all keys in [lo, hi], hot + cold (dedup by key).
void     hc_range_search(HCIndex *idx, BTKey lo, BTKey hi,
                         BTRangeCallback cb, void *arg);

// Get stats snapshot.
HCStats  hc_get_stats(HCIndex *idx);

#endif // HCTREE_H

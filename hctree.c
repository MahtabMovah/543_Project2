// hctree.c
#include "hctree.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

HCIndex* hc_create(int64_t max_key, int btree_degree, HCParams params) {
    HCIndex *idx = (HCIndex*)malloc(sizeof(HCIndex));
    idx->hot  = bt_create(btree_degree);
    idx->cold = bt_create(btree_degree);

    idx->max_key   = max_key;
    idx->hit_score = (double*)calloc((size_t)(max_key + 1), sizeof(double));

    idx->params = params;
    memset(&idx->stats, 0, sizeof(HCStats));

    // Init ML / regression state
    idx->last_q_for_adapt = 0;
    idx->last_hot_nodes   = 0;
    idx->last_cold_nodes  = 0;
    idx->lr_w0            = 0.0;  // bias
    idx->lr_w1            = 0.0;  // weight on D

    return idx;
}

void hc_free(HCIndex *idx) {
    if (!idx) return;
    bt_free(idx->hot);
    bt_free(idx->cold);
    free(idx->hit_score);
    free(idx);
}

void hc_insert(HCIndex *idx, BTKey k, BTPayload v) {
    // For this project, we assume 0 <= k <= max_key.
    if (k < 0 || k > idx->max_key) {
        fprintf(stderr,
                "hc_insert: key %" PRId64 " out of range [0, %" PRId64 "]\n",
                (int64_t)k, (int64_t)idx->max_key);
        return;
    }
    bt_insert(idx->cold, k, v);
}

// --- ML: Online linear regression to adapt sampling rate D ------------
//
// We want to minimize cost(D) = node_visits_per_query(D).
// Model:   cost_hat(D) = w0 + w1 * D
// Update:  SGD on squared loss 0.5 * (cost_hat - cost)^2
//          w0 <- w0 - eta * (cost_hat - cost)
//          w1 <- w1 - eta * (cost_hat - cost) * D
//
// Then pick next D as argmin of cost_hat(D) over [0,1]:
//          D* = clip( -w0 / w1, 0, 1 )
//
static void hc_maybe_adapt_sampling(HCIndex *idx) {
    if (!idx->params.adapt_sampling)
        return;

    const long MIN_DELTA_Q = 5000;   // adapt every 5k queries
    const double ETA       = 0.01;   // learning rate for SGD

    long q  = idx->stats.queries;
    long H  = idx->stats.hot_node_visits;
    long C  = idx->stats.cold_node_visits;

    long dq = q - idx->last_q_for_adapt;
    if (dq < MIN_DELTA_Q)
        return;  // not enough new queries since last update

    long dnodes = (H - idx->last_hot_nodes) + (C - idx->last_cold_nodes);
    if (dq <= 0 || dnodes <= 0) {
        // Avoid weird cases; just update bookkeeping.
        idx->last_q_for_adapt = q;
        idx->last_hot_nodes   = H;
        idx->last_cold_nodes  = C;
        return;
    }

    // Observed cost in this interval
    double cost_interval = (double)dnodes / (double)dq;

    // Current D
    double D = idx->params.sampling_rate;
    if (D < 0.0) D = 0.0;
    if (D > 1.0) D = 1.0;

    // Predicted cost under current model
    double y_hat = idx->lr_w0 + idx->lr_w1 * D;
    double err   = y_hat - cost_interval;  // gradient of squared loss

    // SGD update for linear regression
    idx->lr_w0 -= ETA * err;
    idx->lr_w1 -= ETA * err * D;

    // Choose new D as minimizer of cost_hat(D) = w0 + w1*D.
    // For linear function, minimum on [0,1] is at one of the endpoints
    // or where derivative is zero: d/dD (w0 + w1*D) = w1 => no interior min.
    // So we use a simple heuristic: try to move opposite sign of w1.
    double D_new;
    if (fabs(idx->lr_w1) < 1e-6) {
        // Model is almost flat; keep D unchanged.
        D_new = D;
    } else {
        // If w1 > 0, cost increases with D -> move D down.
        // If w1 < 0, cost decreases with D -> move D up.
        double step = 0.05;
        if (idx->lr_w1 > 0.0)
            D_new = D - step;
        else
            D_new = D + step;
    }

    if (D_new < 0.0) D_new = 0.0;
    if (D_new > 1.0) D_new = 1.0;

    idx->params.sampling_rate = D_new;

    // Update interval bookkeeping
    idx->last_q_for_adapt = q;
    idx->last_hot_nodes   = H;
    idx->last_cold_nodes  = C;
}

// --- Sampling-based promotion ----------------------------------------

static void maybe_promote(HCIndex *idx, BTKey k) {
    if (!idx->params.inclusive) {
        // We only implement inclusive mode in this standalone version.
        return;
    }

    // 1) Sampling-based promotion: with probability D.
    double D = idx->params.sampling_rate;
    if (D < 0.0) D = 0.0;
    if (D > 1.0) D = 1.0;
    double u = (double)rand() / ((double)RAND_MAX + 1.0);
    if (u > D) {
        return; // sampled out, no promotion this time
    }

    // 2) Capacity check: keep hot index under max_hot_fraction of keyspace.
    size_t total_keys = (size_t)(idx->max_key + 1);
    size_t hot_keys   = bt_count_keys(idx->hot);
    size_t cold_keys  = bt_count_keys(idx->cold);
    (void)cold_keys; // not used in inclusive mode

    double max_hot = idx->params.max_hot_fraction * (double)total_keys;
    if ((double)hot_keys >= max_hot) {
        return; // hot index already at capacity
    }

    // 3) If key already in hot, nothing to do.
    BTStats s = {0};
    BTPayload existing = bt_search(idx->hot, k, &s);
    if (existing != NULL) return;

    // 4) Key must exist in cold; fetch payload.
    BTStats s2 = {0};
    BTPayload v = bt_search(idx->cold, k, &s2);
    if (v == NULL) return; // not found; nothing to promote

    bt_insert(idx->hot, k, v);
}

// Point lookup: hot first, then cold.
BTPayload hc_search(HCIndex *idx, BTKey k) {
    idx->stats.queries++;

    // Let the ML controller occasionally update D
    hc_maybe_adapt_sampling(idx);

    BTStats hot_s = {0};
    BTPayload v = bt_search(idx->hot, k, &hot_s);
    idx->stats.hot_node_visits += hot_s.node_visits;

    if (v != NULL) {
        idx->stats.hot_hits++;
        if (k >= 0 && k <= idx->max_key) {
            double old = idx->hit_score[k];
            idx->hit_score[k] = idx->params.decay_alpha * old + 1.0;
            // We don't re-promote; already hot.
        }
        return v;
    }

    BTStats cold_s = {0};
    v = bt_search(idx->cold, k, &cold_s);
    idx->stats.cold_node_visits += cold_s.node_visits;

    if (v != NULL) {
        idx->stats.cold_hits++;
        if (k >= 0 && k <= idx->max_key) {
            double old = idx->hit_score[k];
            double new_score = idx->params.decay_alpha * old + 1.0;
            idx->hit_score[k] = new_score;
            if (new_score >= idx->params.hot_threshold)
                maybe_promote(idx, k);
        }
        return v;
    } else {
        idx->stats.not_found++;
        return NULL;
    }
}

// Helper for deduped range scan: simple callback wrapper
typedef struct {
    BTRangeCallback user_cb;
    void           *user_arg;
    int64_t        *seen;     // seen[key] = 1 if already emitted (size max_key+1)
    HCIndex        *idx;
} HCRangeCtx;

static void hc_range_cb_hot(BTKey k, BTPayload v, void *arg) {
    HCRangeCtx *ctx = (HCRangeCtx*)arg;
    if (k < 0 || k > ctx->idx->max_key) return;
    if (!ctx->seen[k]) {
        ctx->seen[k] = 1;
        ctx->user_cb(k, v, ctx->user_arg);
    }
}

static void hc_range_cb_cold(BTKey k, BTPayload v, void *arg) {
    HCRangeCtx *ctx = (HCRangeCtx*)arg;
    if (k < 0 || k > ctx->idx->max_key) return;
    if (!ctx->seen[k]) {
        ctx->seen[k] = 1;
        ctx->user_cb(k, v, ctx->user_arg);
    }
}

void hc_range_search(HCIndex *idx, BTKey lo, BTKey hi,
                     BTRangeCallback cb, void *arg) {
    HCRangeCtx ctx;
    ctx.user_cb = cb;
    ctx.user_arg = arg;
    ctx.idx = idx;
    ctx.seen = (int64_t*)calloc((size_t)(idx->max_key + 1), sizeof(int64_t));

    BTStats hot_s = {0}, cold_s = {0};
    bt_range_search(idx->hot, lo, hi, hc_range_cb_hot, &ctx, &hot_s);
    bt_range_search(idx->cold, lo, hi, hc_range_cb_cold, &ctx, &cold_s);

    idx->stats.hot_node_visits  += hot_s.node_visits;
    idx->stats.cold_node_visits += cold_s.node_visits;

    free(ctx.seen);
}

HCStats hc_get_stats(HCIndex *idx) {
    HCStats s = idx->stats;
    s.hot_keys  = bt_count_keys(idx->hot);
    s.cold_keys = bt_count_keys(idx->cold);
    return s;
}

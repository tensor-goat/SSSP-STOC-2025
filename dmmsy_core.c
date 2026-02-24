/*
 * dmmsy_core.c – Compiled core for Python DMMSY
 * Compile: gcc -O3 -march=native -shared -fPIC -o libdmmsy.so dmmsy_core.c -lm
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef uint32_t node_t;
typedef double weight_t;

#define WEIGHT_MAX __builtin_inf()
#define NODE_MAX ((node_t)-1)

/* ------------------------------------------------------------------ */
/* 4-Ary Heap (1-indexed)                                             */
/* ------------------------------------------------------------------ */

typedef struct {
    weight_t *val;  /* 1-indexed, size n+1 */
    node_t   *id;   /* 1-indexed, size n+1 */
    node_t   *pos;  /* 0-indexed, size n (0 = absent, NODE_MAX = popped) */
    node_t   *dirty;
    node_t    max_size;
} Heap4;

static inline void heap_push_up(Heap4 *h, node_t i) {
    weight_t nv = h->val[i];
    node_t   ni = h->id[i];
    while (i > 1) {
        node_t par = (i - 2) / 4 + 1;
        if (h->val[par] <= nv) break;
        h->val[i] = h->val[par];
        h->id[i]  = h->id[par];
        h->pos[h->id[par]] = i;
        i = par;
    }
    h->val[i] = nv;
    h->id[i]  = ni;
    h->pos[ni] = i;
}

static inline void heap_push_dec(Heap4 *h, node_t *sz, node_t *dcnt,
                                 node_t n, weight_t d) {
    node_t p = h->pos[n];
    node_t i;
    if (p == 0 || p == NODE_MAX) {
        (*sz)++;
        (*dcnt)++;
        i = *sz;
        h->dirty[*dcnt - 1] = n;
    } else {
        i = p;
        if (d >= h->val[i]) return;
    }
    h->val[i] = d;
    h->id[i]  = n;
    heap_push_up(h, i);
}

static inline void heap_push_down(Heap4 *h, node_t i, node_t sz) {
    weight_t nv = h->val[i];
    node_t   ni = h->id[i];
    for (;;) {
        node_t c1 = i * 4 - 2;
        if (c1 > sz) break;
        node_t mc = c1;
        weight_t mcv = h->val[c1];
        if (c1+1 <= sz && h->val[c1+1] < mcv) { mc = c1+1; mcv = h->val[mc]; }
        if (c1+2 <= sz && h->val[c1+2] < mcv) { mc = c1+2; mcv = h->val[mc]; }
        if (c1+3 <= sz && h->val[c1+3] < mcv) { mc = c1+3; mcv = h->val[mc]; }
        if (nv <= mcv) break;
        h->val[i] = h->val[mc];
        h->id[i]  = h->id[mc];
        h->pos[h->id[mc]] = i;
        i = mc;
    }
    h->val[i] = nv;
    h->id[i]  = ni;
    h->pos[ni] = i;
}

static inline void heap_pop_min(Heap4 *h, node_t *sz, weight_t *mv, node_t *mn) {
    *mv = h->val[1];
    *mn = h->id[1];
    h->pos[*mn] = NODE_MAX;
    if (*sz == 1) { (*sz)--; return; }
    h->val[1] = h->val[*sz];
    h->id[1]  = h->id[*sz];
    h->pos[h->id[1]] = 1;
    (*sz)--;
    heap_push_down(h, 1, *sz);
}

/* ------------------------------------------------------------------ */
/* Parameters                                                         */
/* ------------------------------------------------------------------ */

static inline node_t param_k(node_t n) {
    double l = log2((double)n);
    node_t k = (node_t)floor(pow(l, 1.0/3.0));
    return k < 4 ? 4 : k;
}
static inline node_t param_t(node_t n) {
    double l = log2((double)n);
    node_t t = (node_t)floor(pow(l, 2.0/3.0));
    return t < 2 ? 2 : t;
}

/* ------------------------------------------------------------------ */
/* Workspace (persistent, reused across calls)                        */
/* ------------------------------------------------------------------ */

typedef struct {
    weight_t *d;
    node_t   *pr;
    weight_t *h_val;
    node_t   *h_id;
    node_t   *h_pos;
    node_t   *h_dirty;
    node_t    dh_cnt;
    node_t   *dirty_d;
    node_t    ds_cnt;
    node_t  **piv_bufs;
    node_t    max_depth;
    node_t    n;
} WorkSpace;

static WorkSpace *ws_opt = NULL;
static WorkSpace *ws_res = NULL;

static WorkSpace *alloc_ws(node_t n, node_t k, node_t t) {
    WorkSpace *ws = (WorkSpace *)calloc(1, sizeof(WorkSpace));
    ws->n = n;
    ws->d       = (weight_t *)malloc(sizeof(weight_t) * n);
    ws->pr      = (node_t *)  malloc(sizeof(node_t)   * n);
    ws->h_val   = (weight_t *)malloc(sizeof(weight_t) * (n + 1));
    ws->h_id    = (node_t *)  malloc(sizeof(node_t)   * (n + 1));
    ws->h_pos   = (node_t *)  malloc(sizeof(node_t)   * n);
    ws->h_dirty = (node_t *)  malloc(sizeof(node_t)   * n);
    ws->dirty_d = (node_t *)  malloc(sizeof(node_t)   * n);
    ws->max_depth = t + 2;
    node_t bs = k > 4 ? k : 4;
    ws->piv_bufs = (node_t **)malloc(sizeof(node_t *) * ws->max_depth);
    for (node_t i = 0; i < ws->max_depth; i++)
        ws->piv_bufs[i] = (node_t *)malloc(sizeof(node_t) * bs);
    for (node_t i = 0; i < n; i++) { ws->h_pos[i] = 0; ws->d[i] = WEIGHT_MAX; }
    return ws;
}

static void free_ws(WorkSpace *ws) {
    if (!ws) return;
    for (node_t i = 0; i < ws->max_depth; i++) free(ws->piv_bufs[i]);
    free(ws->piv_bufs);
    free(ws->d); free(ws->pr);
    free(ws->h_val); free(ws->h_id); free(ws->h_pos); free(ws->h_dirty);
    free(ws->dirty_d);
    free(ws);
}

static WorkSpace *get_ws(WorkSpace **slot, node_t n, node_t k, node_t t) {
    if (*slot && (*slot)->n == n) return *slot;
    free_ws(*slot);
    *slot = alloc_ws(n, k, t);
    return *slot;
}

static void reset_ws(WorkSpace *ws, node_t n, node_t src) {
    if (ws->ds_cnt > (n >> 2)) {
        for (node_t i = 0; i < n; i++) { ws->d[i] = WEIGHT_MAX; ws->pr[i] = NODE_MAX; }
    } else {
        for (node_t i = 0; i < ws->ds_cnt; i++) {
            node_t idx = ws->dirty_d[i];
            ws->d[idx] = WEIGHT_MAX;
            ws->pr[idx] = NODE_MAX;
        }
    }
    ws->ds_cnt = 1;
    ws->d[src] = 0.0;
    ws->dirty_d[0] = src;
    ws->dh_cnt = 0;
}

/* ------------------------------------------------------------------ */
/* Dijkstra reference                                                 */
/* ------------------------------------------------------------------ */

void dijkstra_ref(node_t n, const node_t *offset,
                  const node_t *edge_v, const double *edge_w,
                  node_t src, double *d_out, node_t *pr_out) {
    if (n == 0) return;

    for (node_t i = 0; i < n; i++) { d_out[i] = WEIGHT_MAX; pr_out[i] = NODE_MAX; }
    d_out[src] = 0.0;

    /* Allocate heap on stack-ish (malloc for large n) */
    weight_t *hv = (weight_t *)malloc(sizeof(weight_t) * (n + 1));
    node_t   *hi = (node_t *)  malloc(sizeof(node_t)   * (n + 1));
    node_t   *hp = (node_t *)  calloc(n, sizeof(node_t));
    node_t   *hd = (node_t *)  malloc(sizeof(node_t)   * n);

    Heap4 h = { hv, hi, hp, hd, n };
    node_t sz = 0, dcnt = 0;
    heap_push_dec(&h, &sz, &dcnt, src, 0.0);

    while (sz > 0) {
        weight_t du; node_t u;
        heap_pop_min(&h, &sz, &du, &u);
        if (du > d_out[u]) continue;

        node_t u_off = offset[u], u_end = offset[u + 1];
        for (node_t i = u_off; i < u_end; i++) {
            weight_t nd = du + edge_w[i];
            node_t v = edge_v[i];
            if (nd < d_out[v]) {
                d_out[v] = nd;
                pr_out[v] = u;
                heap_push_dec(&h, &sz, &dcnt, v, nd);
            }
        }
    }
    free(hv); free(hi); free(hp); free(hd);
}

/* ------------------------------------------------------------------ */
/* DMMSY Optimised                                                    */
/* ------------------------------------------------------------------ */

static void bmsp_opt(const node_t *offset, const node_t *edge_v,
                     const double *edge_w, node_t gn,
                     node_t *src_buf, node_t off_src, node_t len_src,
                     weight_t B, node_t dp, WorkSpace *ws,
                     node_t k, node_t t) {
    Heap4 h = { ws->h_val, ws->h_id, ws->h_pos, ws->h_dirty, gn };

    if (dp >= t || len_src <= k) {
        /* Reset heap pos for dirty entries */
        for (node_t i = 0; i < ws->dh_cnt; i++) ws->h_pos[ws->h_dirty[i]] = 0;
        ws->dh_cnt = 0;

        node_t sz = 0, dcnt = 0;
        for (node_t i = 0; i < len_src; i++) {
            node_t s = src_buf[off_src + i];
            heap_push_dec(&h, &sz, &dcnt, s, ws->d[s]);
        }
        ws->dh_cnt = dcnt;

        while (sz > 0) {
            weight_t du; node_t u;
            heap_pop_min(&h, &sz, &du, &u);
            if (du > ws->d[u]) continue;

            node_t u_off = offset[u], u_end = offset[u + 1];
            for (node_t i = u_off; i < u_end; i++) {
                weight_t nd = du + edge_w[i];
                node_t v = edge_v[i];
                if (nd < ws->d[v]) {
                    if (ws->d[v] == WEIGHT_MAX)
                        ws->dirty_d[ws->ds_cnt++] = v;
                    ws->d[v] = nd;
                    ws->pr[v] = u;
                    heap_push_dec(&h, &sz, &ws->dh_cnt, v, nd);
                }
            }
        }
        return;
    }

    node_t np = len_src < k ? len_src : k;
    node_t *pivots = ws->piv_bufs[dp + 2];
    node_t step = len_src / np;
    if (step == 0) step = 1;
    node_t curr_np = 0;
    node_t bound = len_src < (step * k) ? len_src : (step * k);
    for (node_t i = 0; i < bound; i += step)
        pivots[curr_np++] = src_buf[off_src + i];

    bmsp_opt(offset, edge_v, edge_w, gn, pivots, 0, curr_np,
             B * 0.5, dp + 1, ws, k, t);

    /* Bounded Dijkstra */
    for (node_t i = 0; i < ws->dh_cnt; i++) ws->h_pos[ws->h_dirty[i]] = 0;
    ws->dh_cnt = 0;

    node_t sz = 0, dcnt = 0;
    int has_work = 0;
    for (node_t i = 0; i < len_src; i++) {
        node_t s = src_buf[off_src + i];
        if (ws->d[s] < B) {
            heap_push_dec(&h, &sz, &dcnt, s, ws->d[s]);
            has_work = 1;
        }
    }
    ws->dh_cnt = dcnt;
    if (!has_work) return;

    while (sz > 0) {
        weight_t du; node_t u;
        heap_pop_min(&h, &sz, &du, &u);
        if (du > ws->d[u]) continue;

        node_t u_off = offset[u], u_end = offset[u + 1];
        for (node_t i = u_off; i < u_end; i++) {
            weight_t nd = du + edge_w[i];
            node_t v = edge_v[i];
            if (nd < ws->d[v]) {
                if (ws->d[v] == WEIGHT_MAX)
                    ws->dirty_d[ws->ds_cnt++] = v;
                ws->d[v] = nd;
                ws->pr[v] = u;
                if (nd < B)
                    heap_push_dec(&h, &sz, &ws->dh_cnt, v, nd);
            }
        }
    }
}

/* Volatile sink to prevent DCE */
static volatile weight_t _sink_w;
static volatile node_t   _sink_n;

/* ------------------------------------------------------------------ */
/* Dijkstra with reused workspace (fair comparison)                   */
/* ------------------------------------------------------------------ */

static WorkSpace *ws_dij = NULL;

void dijkstra_fast(node_t n, const node_t *offset,
                   const node_t *edge_v, const double *edge_w,
                   node_t src, double *d_out, node_t *pr_out) {
    if (n == 0) return;
    node_t k = param_k(n), t = param_t(n);
    WorkSpace *ws = get_ws(&ws_dij, n, k, t);
    reset_ws(ws, n, src);

    Heap4 h = { ws->h_val, ws->h_id, ws->h_pos, ws->h_dirty, n };

    /* Reset stale heap positions */
    for (node_t i = 0; i < ws->dh_cnt; i++) ws->h_pos[ws->h_dirty[i]] = 0;
    ws->dh_cnt = 0;

    node_t sz = 0, dcnt = 0;
    heap_push_dec(&h, &sz, &dcnt, src, 0.0);
    ws->dh_cnt = dcnt;

    while (sz > 0) {
        weight_t du; node_t u;
        heap_pop_min(&h, &sz, &du, &u);
        if (du > ws->d[u]) continue;

        node_t u_off = offset[u], u_end = offset[u + 1];
        for (node_t i = u_off; i < u_end; i++) {
            weight_t nd = du + edge_w[i];
            node_t v = edge_v[i];
            if (nd < ws->d[v]) {
                if (ws->d[v] == WEIGHT_MAX)
                    ws->dirty_d[ws->ds_cnt++] = v;
                ws->d[v] = nd;
                ws->pr[v] = u;
                heap_push_dec(&h, &sz, &ws->dh_cnt, v, nd);
            }
        }
    }

    _sink_w = ws->d[n > 1 ? 1 : 0];
    _sink_n = ws->ds_cnt;

    if (d_out)  memcpy(d_out,  ws->d,  sizeof(weight_t) * n);
    if (pr_out) memcpy(pr_out, ws->pr, sizeof(node_t) * n);
}

void ssp_duan_opt(node_t n, const node_t *offset,
                  const node_t *edge_v, const double *edge_w,
                  double mean_weight,
                  node_t src, double *d_out, node_t *pr_out) {
    if (n == 0) return;
    node_t k = param_k(n), t = param_t(n);
    WorkSpace *ws = get_ws(&ws_opt, n, k, t);
    reset_ws(ws, n, src);

    double log2_n1 = log2((double)(n + 1));
    weight_t B = mean_weight * log2_n1 * 4.0;

    ws->piv_bufs[1][0] = src;
    bmsp_opt(offset, edge_v, edge_w, n, ws->piv_bufs[1], 0, 1, B, 0, ws, k, t);

    /* Prevent compiler from eliminating the computation */
    _sink_w = ws->d[n > 1 ? 1 : 0];
    _sink_n = ws->ds_cnt;

    if (d_out)  memcpy(d_out,  ws->d,  sizeof(weight_t) * n);
    if (pr_out) memcpy(pr_out, ws->pr, sizeof(node_t) * n);
}

/* ------------------------------------------------------------------ */
/* DMMSY Research                                                     */
/* ------------------------------------------------------------------ */

static void bmsp_res(const node_t *offset, const node_t *edge_v,
                     const double *edge_w, node_t gn,
                     node_t *src, node_t src_len,
                     weight_t B, node_t dp, WorkSpace *ws,
                     node_t k, node_t t) {
    Heap4 h = { ws->h_val, ws->h_id, ws->h_pos, ws->h_dirty, gn };

    if (dp >= t || src_len <= k) {
        memset(ws->h_pos, 0, sizeof(node_t) * gn);
        node_t sz = 0, dcnt = 0;
        for (node_t i = 0; i < src_len; i++)
            heap_push_dec(&h, &sz, &dcnt, src[i], ws->d[src[i]]);

        while (sz > 0) {
            weight_t du; node_t u;
            heap_pop_min(&h, &sz, &du, &u);
            if (du > ws->d[u]) continue;
            node_t si = offset[u], ei = offset[u + 1];
            for (node_t i = si; i < ei; i++) {
                weight_t nd = du + edge_w[i];
                node_t v = edge_v[i];
                if (nd <= ws->d[v]) {
                    if (ws->d[v] == WEIGHT_MAX)
                        ws->dirty_d[ws->ds_cnt++] = v;
                    ws->d[v] = nd; ws->pr[v] = u;
                    heap_push_dec(&h, &sz, &dcnt, v, nd);
                }
            }
        }
        return;
    }

    node_t np = src_len < k ? src_len : k;
    node_t *pivots = ws->piv_bufs[dp + 2];
    node_t step = src_len / np;
    if (step == 0) step = 1;
    node_t bound = src_len < (step * k) ? src_len : (step * k);
    node_t curr_np = 0;
    for (node_t i = 0; i < bound; i += step)
        pivots[curr_np++] = src[i];

    bmsp_res(offset, edge_v, edge_w, gn, pivots, curr_np,
             B * 0.5, dp + 1, ws, k, t);

    memset(ws->h_pos, 0, sizeof(node_t) * gn);
    node_t sz = 0, dcnt = 0;
    for (node_t i = 0; i < src_len; i++) {
        node_t s = src[i];
        if (ws->d[s] < B)
            heap_push_dec(&h, &sz, &dcnt, s, ws->d[s]);
    }

    while (sz > 0) {
        weight_t du; node_t u;
        heap_pop_min(&h, &sz, &du, &u);
        if (du > ws->d[u]) continue;
        node_t si = offset[u], ei = offset[u + 1];
        for (node_t i = si; i < ei; i++) {
            weight_t nd = du + edge_w[i];
            node_t v = edge_v[i];
            if (nd <= ws->d[v]) {
                if (ws->d[v] == WEIGHT_MAX)
                    ws->dirty_d[ws->ds_cnt++] = v;
                ws->d[v] = nd; ws->pr[v] = u;
                if (nd < B)
                    heap_push_dec(&h, &sz, &dcnt, v, nd);
            }
        }
    }
}

void ssp_duan_res(node_t n, const node_t *offset,
                  const node_t *edge_v, const double *edge_w,
                  double mean_weight,
                  node_t src, double *d_out, node_t *pr_out) {
    if (n == 0) return;
    node_t k = param_k(n), t = param_t(n);
    WorkSpace *ws = get_ws(&ws_res, n, k, t);
    reset_ws(ws, n, src);

    double log2_n1 = log2((double)(n + 1));
    weight_t B = mean_weight * log2_n1 * 4.0;

    ws->piv_bufs[0][0] = src;
    bmsp_res(offset, edge_v, edge_w, n, ws->piv_bufs[0], 1, B, 0, ws, k, t);

    _sink_w = ws->d[n > 1 ? 1 : 0];
    _sink_n = ws->ds_cnt;

    if (d_out)  memcpy(d_out,  ws->d,  sizeof(weight_t) * n);
    if (pr_out) memcpy(pr_out, ws->pr, sizeof(node_t) * n);
}

#include "ggml.h"
#include "ggml-alloc.h"
#include "llama.h"
#include <unordered_map>
#include <vector>
#include <cassert>
#include <climits>
#include <cstring>
#include <cstdarg>
#include <ctime>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <string>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static const float rms_norm_eps = LLAMA_DEFAULT_RMS_EPS;

struct random_normal_distribution {
    std::mt19937 gen;
    std::normal_distribution<float> rd;
    float min;
    float max;
};

struct random_uniform_distribution {
    std::mt19937 gen;
    std::uniform_real_distribution<float> rd;
};

void init_random_normal_distribution(struct random_normal_distribution * rnd, int seed, float mean, float std, float min, float max) {
    rnd->gen = std::mt19937(seed);
    rnd->rd = std::normal_distribution<float>{mean, std};
    rnd->min = min;
    rnd->max = max;
}

void init_random_uniform_distribution(struct random_uniform_distribution * rnd, int seed, float min, float max) {
    rnd->gen = std::mt19937(seed);
    rnd->rd = std::uniform_real_distribution<float>{min, max};
}

int clamp(const int v, const int min, const int max) {
    return ((v < min) ? (min) : (v > max) ? (max) : v);
}

float fclamp(const float v, const float min, const float max) {
    return ((v < min) ? (min) : (v > max) ? (max) : v);
}

float frand() {
    return (float)rand()/(float)RAND_MAX;
}

float frand_normal(struct random_normal_distribution * rnd) {
    return fclamp(rnd->rd(rnd->gen), rnd->min, rnd->max);
}

float frand_uniform(struct random_uniform_distribution * rnd) {
    return rnd->rd(rnd->gen);
}

void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

struct ggml_tensor * randomize_tensor_normal(struct ggml_tensor * tensor, struct random_normal_distribution * rnd) {
    float scale = 1.0f; // xavier
    switch (tensor->n_dims) {
        case 1:
            scale /= sqrtf(tensor->ne[0]);
            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0]);
                *dst = scale * frand_normal(rnd);
            }
            break;
        case 2:
            scale /= sqrtf(tensor->ne[0]+tensor->ne[1]);
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
                    *dst = scale * frand_normal(rnd);
                }
            }
            break;
        case 3:
            scale /= sqrtf(tensor->ne[0]+tensor->ne[1]);
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                        float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2]);
                        *dst = scale * frand_normal(rnd);
                    }
                }
            }
            break;
        case 4:
            scale /= sqrtf(tensor->ne[0]+tensor->ne[1]);
            for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
                for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                    for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                        for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                            float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3]);
                            *dst = scale * frand_normal(rnd);
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    };
    return tensor;
}

struct ggml_tensor * randomize_tensor_uniform(struct ggml_tensor * tensor, struct random_uniform_distribution * rnd) {
    switch (tensor->n_dims) {
        case 1:
            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0]);
                *dst = frand_uniform(rnd);
            }
            break;
        case 2:
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
                    *dst = frand_uniform(rnd);
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                        float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2]);
                        *dst = frand_uniform(rnd);
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
                for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                    for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                        for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                            float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3]);
                            *dst = frand_uniform(rnd);
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    };
    return tensor;
}

struct llama_vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};

struct my_llama_hparams {
    uint32_t n_vocab = 32000;
    uint32_t n_ctx   = 512;   // this is provided as user input?
    uint32_t n_embd  = 4096;
    uint32_t n_mult  = 4;
    uint32_t n_head  = 32;
    uint32_t n_layer = 32;
    uint32_t n_rot   = 64;

    bool operator!=(const my_llama_hparams& other) const {
        return memcmp(this, &other, sizeof(my_llama_hparams));
    }
};

struct my_llama_layer {
    // normalization
    struct ggml_tensor * attention_norm;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;

    // normalization
    struct ggml_tensor * ffn_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;
};

struct my_llama_kv_cache {
    struct ggml_context * ctx = NULL;

    struct ggml_tensor * k;
    struct ggml_tensor * v;

    // llama_ctx_buffer buf;

    int n; // number of tokens currently in the cache
};

struct my_llama_model {
    struct ggml_context * ctx = NULL;

    my_llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<my_llama_layer> layers;

    uint32_t train_its = 0;
    uint32_t train_samples = 0;
    uint32_t train_tokens = 0;
};

uint32_t get_n_ff(const struct my_llama_hparams* hparams) {
    const uint32_t n_ff = ((2*(4*hparams->n_embd)/3 + hparams->n_mult - 1)/hparams->n_mult)*hparams->n_mult;
    return n_ff;
}

void print_params(struct my_llama_hparams * params) {
    printf("%s: n_vocab: %d\n", __func__, params->n_vocab);
    printf("%s: n_ctx:   %d\n", __func__, params->n_ctx);
    printf("%s: n_embd:  %d\n", __func__, params->n_embd);
    printf("%s: n_mult:  %d\n", __func__, params->n_mult);
    printf("%s: n_head:  %d\n", __func__, params->n_head);
    printf("%s: n_ff:    %d\n", __func__, get_n_ff(params));
    printf("%s: n_layer: %d\n", __func__, params->n_layer);
    printf("%s: n_rot:   %d\n", __func__, params->n_rot);
}

void init_model(struct my_llama_model * model) {
    const auto & hparams = model->hparams;

    const uint32_t n_embd  = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_vocab = hparams.n_vocab;

    const uint32_t n_ff = get_n_ff(&hparams);

    struct ggml_context * ctx = model->ctx;

    model->train_its = 0;
    model->train_samples = 0;
    model->train_tokens = 0;

    model->tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    model->norm           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    model->output         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);

    ggml_set_name(model->tok_embeddings, "tok_embeddings.weight");
    ggml_set_name(model->norm,           "norm.weight");
    ggml_set_name(model->output,         "output.weight");

    model->layers.resize(n_layer);
    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];

        std::string layers_i = "layers." + std::to_string(i);

        layer.attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        layer.wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        layer.wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        layer.wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        layer.wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);

        layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        layer.w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);
        layer.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,   n_ff, n_embd);
        layer.w3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);

        ggml_set_name(layer.attention_norm, (layers_i + ".attention_norm.weight").c_str());

        ggml_set_name(layer.wq, (layers_i + ".attention.wq.weight").c_str());
        ggml_set_name(layer.wk, (layers_i + ".attention.wk.weight").c_str());
        ggml_set_name(layer.wv, (layers_i + ".attention.wv.weight").c_str());
        ggml_set_name(layer.wo, (layers_i + ".attention.wo.weight").c_str());

        ggml_set_name(layer.ffn_norm, (layers_i + ".ffn_norm.weight").c_str());

        ggml_format_name(layer.w1, "%s.feed_forward.w1.weight", layers_i.c_str());
        ggml_format_name(layer.w2, "%s.feed_forward.w2.weight", layers_i.c_str());
        ggml_format_name(layer.w3, "%s.feed_forward.w3.weight", layers_i.c_str());
    }
}

void set_param_model(struct my_llama_model * model) {
    const auto& hparams = model->hparams;

    const uint32_t n_layer = hparams.n_layer;

    struct ggml_context* ctx = model->ctx;

    ggml_set_param(ctx, model->tok_embeddings);
    ggml_set_param(ctx, model->norm);
    ggml_set_param(ctx, model->output);

    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];

        ggml_set_param(ctx, layer.attention_norm);
        ggml_set_param(ctx, layer.wq);
        ggml_set_param(ctx, layer.wk);
        ggml_set_param(ctx, layer.wv);
        ggml_set_param(ctx, layer.wo);
        ggml_set_param(ctx, layer.ffn_norm);
        ggml_set_param(ctx, layer.w1);
        ggml_set_param(ctx, layer.w2);
        ggml_set_param(ctx, layer.w3);
    }
}

void randomize_model(struct my_llama_model * model, int seed, float mean, float std, float min, float max) {
    const auto & hparams = model->hparams;

    const uint32_t n_layer = hparams.n_layer;

    struct random_normal_distribution rnd;
    init_random_normal_distribution(&rnd, seed, mean, std, min, max);

    randomize_tensor_normal(model->tok_embeddings, &rnd);
    randomize_tensor_normal(model->norm,           &rnd);
    randomize_tensor_normal(model->output,         &rnd);

    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];
        randomize_tensor_normal(layer.attention_norm, &rnd);

        randomize_tensor_normal(layer.wq, &rnd);
        randomize_tensor_normal(layer.wk, &rnd);
        randomize_tensor_normal(layer.wv, &rnd);
        randomize_tensor_normal(layer.wo, &rnd);

        randomize_tensor_normal(layer.ffn_norm, &rnd);

        randomize_tensor_normal(layer.w1, &rnd);
        randomize_tensor_normal(layer.w2, &rnd);
        randomize_tensor_normal(layer.w3, &rnd);
    }
}

bool init_kv_cache(struct my_llama_kv_cache* cache, struct my_llama_model * model, int n_batch) {
    const auto & hparams = model->hparams;

    const uint32_t n_ctx   = hparams.n_ctx;
    const uint32_t n_embd  = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;

    const int64_t n_mem      = n_layer*n_ctx*n_batch;
    const int64_t n_elements = n_embd*n_mem;

    // cache.buf.resize(2u*n_elements*ggml_type_size(wtype) + 2u*MB);

    // struct ggml_init_params params;
    // params.mem_size   = cache.buf.size;
    // params.mem_buffer = cache.buf.addr;
    // params.no_alloc   = false;
    if (!cache->ctx) {
        struct ggml_init_params params;
        params.mem_size   = 2u*n_elements*ggml_type_size(GGML_TYPE_F32) + 2u*1024*1024;
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        cache->ctx = ggml_init(params);

        if (!cache->ctx) {
            fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
            return false;
        }
    }

    cache->k = ggml_new_tensor_1d(cache->ctx, GGML_TYPE_F32, n_elements);
    cache->v = ggml_new_tensor_1d(cache->ctx, GGML_TYPE_F32, n_elements);

    return true;
}

struct ggml_tensor * forward(
        struct my_llama_model    * model,
        struct my_llama_kv_cache * cache,
        struct ggml_context   * ctx0,
        struct ggml_cgraph    * gf,
        struct ggml_tensor    * tokens_input,
        const  int              n_tokens,
        const  int              n_past) {

    const int N = n_tokens;

    struct my_llama_kv_cache& kv_self = *cache;
    const auto & hparams = model->hparams;
    const int n_ctx   = hparams.n_ctx;
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_head  = hparams.n_head;
    const int n_rot   = hparams.n_rot;

    struct ggml_tensor * tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(tokens->data, tokens_input->data, N*ggml_element_size(tokens));

    struct ggml_tensor * kc = kv_self.k;
    struct ggml_tensor * vc = kv_self.v;

    // inpL shape [n_embd,N,1,1]
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model->tok_embeddings, tokens);
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        struct ggml_tensor * cur;

        // lctx.use_buf(ctx0, 0);

        // norm
        {
            // cur shape [n_embd,N,1,1]
            cur = ggml_rms_norm(ctx0, inpL, rms_norm_eps);

            // cur = attention_norm*cur
            cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].attention_norm, cur),
                        cur);
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            // wq   shape [n_embd, n_embd, 1, 1]
            // wk   shape [n_embd, n_embd, 1, 1]
            // Qcur shape [n_embd/n_head, n_head, N, 1]
            // Kcur shape [n_embd/n_head, n_head, N, 1]
            struct ggml_tensor * Qcur = ggml_rope_inplace(ctx0, ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wq, cur), n_embd/n_head, n_head, N), n_past, n_rot, 0, n_ctx);
            struct ggml_tensor * Kcur = ggml_rope_inplace(ctx0, ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wk, cur), n_embd/n_head, n_head, N), n_past, n_rot, 0, n_ctx);

            // store key and value to memory
            {
                // compute the transposed [N, n_embd] V matrix
                // wv   shape [n_embd, n_embd, 1, 1]
                // Vcur shape [n_embd, N, 1, 1]
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wv, cur), n_embd, N)));

                // kv_self.k shape [n_embd * n_ctx * n_layer, 1]
                // kv_self.v shape [n_embd * n_ctx * n_layer, 1]
                // k         shape [n_embd * N, 1]   == kv_self.k[:,n_past:n_past+N,il,0]
                // v         shape [N, n_embd, 1, 1] == kv_self.v[:,n_past:n_past+N,il,0]

                /* {
                    struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd, (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
                    struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, N, n_embd,
                            (   n_ctx)*ggml_element_size(kv_self.v),
                            (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd + n_past*ggml_element_size(kv_self.v));

                    // important: storing RoPE-ed version of K in the KV cache!
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
                } //*/

                kc = ggml_set_1d_inplace(ctx0, kc, ggml_reshape_1d(ctx0, Kcur, n_embd*N), (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
                vc = ggml_set_2d_inplace(ctx0, vc, Vcur, (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd + n_past*ggml_element_size(kv_self.v));
            }

            // Qcur shape [n_embd/n_head, n_head, N, 1]
            // Q shape    [n_embd/n_head, N, n_head, 1]
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);

            // kv_self.k shape [n_embd * n_ctx * n_layer, 1]
            // K shape [n_embd/n_head, n_past + N, n_head, 1]
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, kc, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(kc)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // K * Q
            // KQ shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // KQ_scaled shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_scaled =
                ggml_scale(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrtf(float(n_embd)/n_head)));

            // KQ_masked = mask_past(KQ_scaled)
            // KQ_masked shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            // KQ_soft_max shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // split cached V into n_head heads
            //// V shape [n_past + N, n_embd/n_head, n_head, 1]
            // V shape [n_past + N, n_embd/n_head, n_head, 1] == kv_self.v[:,:(n_past+N),il,1]
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, vc,
                        n_past + N, n_embd/n_head, n_head,
                        n_ctx*ggml_element_size(vc),
                        n_ctx*ggml_element_size(vc)*n_embd/n_head,
                        il*n_ctx*ggml_element_size(vc)*n_embd);

            // KQV shape [n_embd/n_head, N, n_head, 1]
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // KQV_merged shape [n_embd/n_head, n_head, N, 1]
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            // KQV_merged shape

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // cur shape [n_embd,N,1,1]
            cur = ggml_reshape_2d(ctx0, ggml_cont(ctx0, KQV_merged), n_embd, N);
            // cur = ggml_cpy(ctx0,
            //         KQV_merged,
            //         ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (no bias)
            // cur shape [n_embd,N,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].wo,
                    cur);
        }

        // lctx.use_buf(ctx0, 1);

        // inpFF shape [n_embd,N,1,1]
        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);

        // feed-forward network
        {
            // norm
            {
                // cur shape [n_embd,N,1,1]
                cur = ggml_rms_norm(ctx0, inpFF, rms_norm_eps);

                // cur = ffn_norm*cur
                // cur shape [n_embd,N,1,1]
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].ffn_norm, cur),
                        cur);
            }

            // tmp shape [n_ff,N,1,1]
            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model->layers[il].w3,
                    cur);

            // cur shape [n_ff,N,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w1,
                    cur);

            // SILU activation
            // cur shape [n_ff,N,1,1]
            cur = ggml_silu(ctx0, cur);

            // cur shape [n_ff,N,1,1]
            cur = ggml_mul(ctx0, cur, tmp);

            // cur shape [n_embd,N,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w2,
                    cur);
        }

        // cur shape [n_embd,N,1,1]
        cur = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        // inpL shape [n_embd,N,1,1]
        inpL = cur;
    }

    // norm
    {

        // inpL shape [n_embd,N,1,1]
        inpL = ggml_rms_norm(ctx0, inpL, rms_norm_eps);

        // inpL = norm*inpL
        // inpL shape [n_embd,N,1,1]
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, model->norm, inpL),
                    inpL);

        //embeddings = inpL;
    }

    // lm_head
    // inpL shape [n_vocab,N,1,1]
    inpL = ggml_mul_mat(ctx0, model->output, inpL);

    // run the computation
    ggml_build_forward_expand(gf, inpL);

    return inpL;
}

void assert_shape_1d(struct ggml_tensor * tensor, int64_t ne0) {
    GGML_ASSERT(tensor->n_dims == 1);
    GGML_ASSERT(tensor->ne[0] == ne0);
}

void assert_shape_2d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    GGML_ASSERT(tensor->n_dims == 2);
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
}

void assert_shape_3d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2) {
    GGML_ASSERT(tensor->n_dims == 3);
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
    GGML_ASSERT(tensor->ne[2] == ne2);
}

void assert_shape_4d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    GGML_ASSERT(tensor->n_dims == 4);
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
    GGML_ASSERT(tensor->ne[2] == ne2);
    GGML_ASSERT(tensor->ne[3] == ne3);
}

static size_t hash(void * p) {
    return (size_t)p % GGML_GRAPH_HASHTABLE_SIZE;
}

static size_t hash_find(void * hash_table[], void * p) {
    size_t h = hash(p);

    // linear probing
    size_t i = h;
    while (hash_table[i] != NULL && hash_table[i] != p) {
        i = (i + 1) % GGML_GRAPH_HASHTABLE_SIZE;
        if (i == h) {
            // visited all hash table entries -> not found
            return GGML_GRAPH_HASHTABLE_SIZE;
        }
    }
    return i;
}

static bool hash_insert(void * hash_table[], void * p) {
    size_t h = hash(p);
    size_t i = hash_find(hash_table, p);

    GGML_ASSERT(i < GGML_GRAPH_HASHTABLE_SIZE); // assert that not full

    if (hash_table[i] == p) {
        return true;
    }

    // insert
    GGML_ASSERT(hash_table[i] == NULL);
    hash_table[i] = p;
    return false;
}

static bool hash_contains(void * hash_table[], void * p) {
    size_t i = hash_find(hash_table, p);
    return (i < GGML_GRAPH_HASHTABLE_SIZE) && (hash_table[i] == p);
}

struct hash_map {
    void * keys[GGML_GRAPH_HASHTABLE_SIZE];
    void * vals[GGML_GRAPH_HASHTABLE_SIZE];
};
static const size_t HASH_MAP_SIZE = sizeof(struct hash_map);

struct hash_map * new_hash_map() {
    struct hash_map * result = new struct hash_map;
    for (int i=0; i<GGML_GRAPH_HASHTABLE_SIZE; ++i) {
        result->keys[i] = NULL;
        result->vals[i] = NULL;
    }
    return result;
};

void free_hash_map(struct hash_map * map) {
    delete map;
}

static bool ggml_is_view(struct ggml_tensor * t) {
    return t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_PERMUTE || t->op == GGML_OP_CPY;
}

static struct ggml_tensor * get_view_parent(struct ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_PERMUTE:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_VIEW:
            return t->src[0];
        case GGML_OP_CPY:
            return t->src[1];
        default:
            return NULL;
    }
}

static struct ggml_tensor * get_view_source(struct ggml_tensor * t) {
    struct ggml_tensor * parent = t;
    do {
        parent = get_view_parent(parent);
    } while (ggml_is_view(parent));
    return parent;
}

struct ggml_tensor * ggml_recompute_graph_node(
        struct ggml_context * ctx,
        struct ggml_cgraph  * graph,
        struct hash_map     * replacements,
        struct ggml_tensor  * node) {

    if (node == NULL) {
        return NULL;
    }

    if (node->is_param) {
        return node;
    }

    if (!hash_contains(graph->visited_hash_table, node)) {
        return node;
    }

    int count_children = 0;
    for (int k = 0; k < GGML_MAX_SRC; ++k) {
        if (node->src[k]) {
            ++count_children;
        }
    }

    if (count_children == 0) {
        return node;
    }

    size_t i = hash_find(replacements->keys, node);
    GGML_ASSERT(i < GGML_GRAPH_HASHTABLE_SIZE); // assert that not full
    if (replacements->keys[i] == node) {
        return (struct ggml_tensor *) replacements->vals[i];
    }

    struct ggml_tensor * clone = ggml_new_tensor(ctx, node->type, node->n_dims, node->ne);

    // insert clone into replacements
    GGML_ASSERT(replacements->keys[i] == NULL); // assert that we don't overwrite
    replacements->keys[i] = node;
    replacements->vals[i] = clone;

    clone->op       = node->op;
    clone->grad     = node->grad;
    clone->is_param = node->is_param;
    clone->extra    = node->extra;
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        clone->nb[k] = node->nb[k];
    }
    for (int k = 0; k < GGML_MAX_SRC; ++k) {
        clone->src[k] = ggml_recompute_graph_node(ctx, graph, replacements, node->src[k]);
    }
    if (ggml_is_view(clone)) {
        struct ggml_tensor * source = get_view_source(clone);
        GGML_ASSERT(source != NULL);
        clone->data = source->data;
    }

    GGML_ASSERT(sizeof(node->op_params) == sizeof(int32_t) * (GGML_MAX_OP_PARAMS / sizeof(int32_t)));
    GGML_ASSERT(sizeof(node->name)      == GGML_MAX_NAME);
    memcpy(clone->op_params, node->op_params, sizeof(node->op_params));
    ggml_format_name(clone, "%s (clone)", ggml_get_name(node));
    printf("%s: new clone: op=%s name=%s\n", __func__, ggml_op_name(clone->op), ggml_get_name(clone));
    // memcpy(clone->name,      node->name,      sizeof(node->name));

    return clone;
};

void ggml_build_backward_gradient_checkpointing(
        struct ggml_context   * ctx,
        struct ggml_cgraph    * gf,
        struct ggml_cgraph    * gb,
        struct ggml_cgraph    * gb_tmp,
        struct ggml_tensor  * * checkpoints,
        int                     n_checkpoints) {
    *gb_tmp = *gf;
    ggml_build_backward_expand(ctx, gf, gb_tmp, true);

    if (n_checkpoints <= 0) {
        *gb = *gb_tmp;
        return;
    }

    struct hash_map * replacements = new_hash_map();

    // insert checkpoints in replacements
    for (int i = 0; i < n_checkpoints; ++i) {
        size_t k = hash_find(replacements->keys, checkpoints[i]);
        GGML_ASSERT(k < GGML_GRAPH_HASHTABLE_SIZE); // assert that not full
        GGML_ASSERT(replacements->keys[k] == NULL); // assert that we don't overwrite
        replacements->keys[k] = checkpoints[i];
        replacements->vals[k] = checkpoints[i];
    }

    *gb = *gf;
    // rewrite gb_tmp->nodes[gf->n_nodes:gb_tmp->n_nodes],
    // replacing references to gb_tmp->nodes[0:gf->n_nodes] ( == gf->nodes[0:gf->n_nodes]),
    // by recomputing them from checkpoints
    for (int i = gf->n_nodes; i<gb_tmp->n_nodes; ++i) {
        struct ggml_tensor * node = gb_tmp->nodes[i];
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            // insert new tensors recomputing src, reusing already made replacements,
            // remember replacements: remember new tensors with mapping from corresponding gf nodes
            // recurse for input tensors,
            // unless (i.e. terminating when) input tensors are checkpoints
            node->src[k] = ggml_recompute_graph_node(ctx, gf, replacements, node->src[k]);
        }
        // insert rewritten backward node with replacements made into resulting backward graph gb
        ggml_build_forward_expand(gb, node);
    }

    free_hash_map(replacements);
}

struct ggml_tensor * llama_build_train_graphs(
        struct my_llama_model * model,
        struct ggml_allocr    * alloc,
        struct ggml_context   * ctx,
        struct ggml_cgraph    * gf,
        struct ggml_cgraph    * gb,
        struct ggml_cgraph    * gb_tmp,
        struct ggml_tensor  * * logits,
        struct ggml_tensor    * tokens_input,
        struct ggml_tensor    * targets,
        const  int              n_tokens,
        const  int              n_batch,
        const  bool             enable_flash_attn,
        const  bool             enable_checkpointing) {

    ggml_set_scratch(ctx, { 0, 0, nullptr, });
    const int n_past = 0;
    const int N = n_tokens;
    const auto & hparams = model->hparams;
    const int n_ctx      = hparams.n_ctx;
    const int n_vocab    = hparams.n_vocab;
    const int n_embd     = hparams.n_embd;
    const int n_layer    = hparams.n_layer;
    const int n_head     = hparams.n_head;
    const int n_rot      = hparams.n_rot;
    const int n_ff       = get_n_ff(&hparams);
    const int rope_mode  = 0;

    auto set_name = [](struct ggml_tensor * t, const char * n) {
        ggml_set_name(t, n);
        if (t->grad) {
            ggml_format_name(t->grad, "%s->grad", n);
        }
    };

    set_name(tokens_input, "tokens_input");
    set_name(targets,      "targets");

    GGML_ASSERT(tokens_input->type == GGML_TYPE_I32);
    struct ggml_tensor * t00 = ggml_reshape_1d(ctx, tokens_input, N*n_batch);  set_name(t00, "t00"); assert_shape_1d(t00, N*n_batch);
    struct ggml_tensor * t01 = ggml_get_rows(ctx, model->tok_embeddings, t00); set_name(t01, "t01"); assert_shape_2d(t01, n_embd, N*n_batch);

    struct ggml_tensor * cur = t01;

    std::vector<struct ggml_tensor *> checkpoints;
    checkpoints.push_back(tokens_input);
    checkpoints.push_back(targets);
    checkpoints.push_back(t00);
    checkpoints.push_back(t01);

    struct ggml_tensor * kv_scale;
    if (!enable_flash_attn) {
        kv_scale = ggml_new_f32(ctx, 1.0f/sqrtf(float(n_embd)/n_head));
    }

    for (int il = 0; il < n_layer; ++il) {
        struct my_llama_layer & layer = model->layers[il];
        struct ggml_tensor * t02 = ggml_rms_norm     (ctx, cur, rms_norm_eps);                      set_name(t02, "t02");     assert_shape_2d(t02, n_embd, N*n_batch);
        struct ggml_tensor * t03 = ggml_repeat       (ctx, layer.attention_norm, t02);              set_name(t03, "t03");     assert_shape_2d(t03, n_embd, N*n_batch);
        struct ggml_tensor * t04 = ggml_mul          (ctx, t03, t02);                               set_name(t04, "t04");     assert_shape_2d(t04, n_embd, N*n_batch);
        struct ggml_tensor * t05 = ggml_mul_mat      (ctx, layer.wq, t04);                          set_name(t05, "t05");     assert_shape_2d(t05, n_embd, N*n_batch);
        struct ggml_tensor * t06 = ggml_reshape_4d   (ctx, t05, n_embd/n_head, n_head, N, n_batch); set_name(t06, "t06");     assert_shape_4d(t06, n_embd/n_head, n_head, N, n_batch);
        struct ggml_tensor * t07 = ggml_rope_inplace (ctx, t06, n_past, n_rot, rope_mode, n_ctx);   set_name(t07, "t07");     assert_shape_4d(t07, n_embd/n_head, n_head, N, n_batch);
        struct ggml_tensor * t08 = ggml_mul_mat      (ctx, layer.wk, t04);                          set_name(t08, "t08");     assert_shape_2d(t08, n_embd, N*n_batch);
        struct ggml_tensor * t09 = ggml_reshape_4d   (ctx, t08, n_embd/n_head, n_head, N, n_batch); set_name(t09, "t09");     assert_shape_4d(t09, n_embd/n_head, n_head, N, n_batch);
        struct ggml_tensor * t10 = ggml_rope_inplace (ctx, t09, n_past, n_rot, rope_mode, n_ctx);   set_name(t10, "t10");     assert_shape_4d(t10, n_embd/n_head, n_head, N, n_batch);
        struct ggml_tensor * t11 = ggml_mul_mat      (ctx, t04, layer.wv);                          set_name(t11, "t11");     assert_shape_2d(t11, N*n_batch, n_embd);
        struct ggml_tensor * t12 = ggml_reshape_4d   (ctx, t11, N, n_batch, n_embd/n_head, n_head); set_name(t12, "t12");     assert_shape_4d(t12, N, n_batch, n_embd/n_head, n_head);
        struct ggml_tensor * t13 = ggml_permute      (ctx, t07, 0, 2, 1, 3);                        set_name(t13, "t13");     assert_shape_4d(t13, n_embd/n_head, N, n_head, n_batch);
        struct ggml_tensor * t14 = ggml_permute      (ctx, t10, 0, 2, 1, 3);                        set_name(t14, "t14");     assert_shape_4d(t14, n_embd/n_head, N, n_head, n_batch);
        struct ggml_tensor * t15 = ggml_permute      (ctx, t12, 0, 3, 1, 2);                        set_name(t15, "t15");     assert_shape_4d(t15, N, n_embd/n_head, n_head, n_batch);
        struct ggml_tensor * t16;
        if (enable_flash_attn) {
            t16 = ggml_flash_attn(ctx, t13, t14, t15, true);                                        set_name(t16, "t16");     assert_shape_4d(t16, n_embd/n_head, N, n_head, n_batch);
        } else {
            struct ggml_tensor * t16_0 = ggml_mul_mat              (ctx, t14, t13);                 set_name(t16_0, "t16_0"); assert_shape_4d(t16_0, N, N, n_head, n_batch);
            struct ggml_tensor * t16_1 = ggml_scale_inplace        (ctx, t16_0, kv_scale);          set_name(t16_1, "t16_1"); assert_shape_4d(t16_1, N, N, n_head, n_batch);
            struct ggml_tensor * t16_2 = ggml_diag_mask_inf_inplace(ctx, t16_1, n_past);            set_name(t16_2, "t16_2"); assert_shape_4d(t16_2, N, N, n_head, n_batch);
            struct ggml_tensor * t16_3 = ggml_soft_max_inplace     (ctx, t16_2);                    set_name(t16_3, "t16_3"); assert_shape_4d(t16_3, N, N, n_head, n_batch);
            t16 = ggml_mul_mat(ctx, t15, t16_3);                                                    set_name(t16, "t16");     assert_shape_4d(t16, n_embd/n_head, N, n_head, n_batch);
        }
        struct ggml_tensor * t17 = ggml_permute      (ctx, t16, 0, 2, 1, 3);                        set_name(t17, "t17");     assert_shape_4d(t17, n_embd/n_head, n_head, N, n_batch);
        struct ggml_tensor * t18 = ggml_cont         (ctx, t17);                                    set_name(t18, "t18");     assert_shape_4d(t18, n_embd/n_head, n_head, N, n_batch);
        struct ggml_tensor * t19 = ggml_reshape_2d   (ctx, t18, n_embd, N*n_batch);                 set_name(t19, "t19");     assert_shape_2d(t19, n_embd, N*n_batch);
        struct ggml_tensor * t20 = ggml_mul_mat      (ctx, layer.wo, t19);                          set_name(t20, "t20");     assert_shape_2d(t20, n_embd, N*n_batch);
        struct ggml_tensor * t21 = ggml_add          (ctx, t20, cur);                               set_name(t21, "t21");     assert_shape_2d(t21, n_embd, N*n_batch);
        struct ggml_tensor * t22 = ggml_rms_norm     (ctx, t21, rms_norm_eps);                      set_name(t22, "t22");     assert_shape_2d(t22, n_embd, N*n_batch);
        struct ggml_tensor * t23 = ggml_repeat       (ctx, layer.ffn_norm, t22);                    set_name(t23, "t23");     assert_shape_2d(t23, n_embd, N*n_batch);
        struct ggml_tensor * t24 = ggml_mul          (ctx, t23, t22);                               set_name(t24, "t24");     assert_shape_2d(t24, n_embd, N*n_batch);
        struct ggml_tensor * t25 = ggml_mul_mat      (ctx, layer.w3, t24);                          set_name(t25, "t25");     assert_shape_2d(t25, n_ff, N*n_batch);
        struct ggml_tensor * t26 = ggml_mul_mat      (ctx, layer.w1, t24);                          set_name(t26, "t26");     assert_shape_2d(t26, n_ff, N*n_batch);
        struct ggml_tensor * t27 = ggml_silu         (ctx, t26);                                    set_name(t27, "t27");     assert_shape_2d(t27, n_ff, N*n_batch);
        struct ggml_tensor * t28 = ggml_mul          (ctx, t27, t25);                               set_name(t28, "t28");     assert_shape_2d(t28, n_ff, N*n_batch);
        struct ggml_tensor * t29 = ggml_mul_mat      (ctx, layer.w2, t28);                          set_name(t29, "t29");     assert_shape_2d(t29, n_embd, N*n_batch);
        struct ggml_tensor * t30 = ggml_add          (ctx, t29, t21);                               set_name(t30, "t30");     assert_shape_2d(t30, n_embd, N*n_batch);
        cur = t30;
        checkpoints.push_back(cur);
    }
    struct ggml_tensor * t31   = ggml_rms_norm          (ctx, cur, rms_norm_eps);                   set_name(t31, "t31");     assert_shape_2d(t31, n_embd, N*n_batch);
    struct ggml_tensor * t32   = ggml_repeat            (ctx, model->norm, t31);                    set_name(t32, "t32");     assert_shape_2d(t32, n_embd, N*n_batch);
    struct ggml_tensor * t33   = ggml_mul               (ctx, t32, t31);                            set_name(t33, "t33");     assert_shape_2d(t33, n_embd, N*n_batch);
    struct ggml_tensor * t34   = ggml_mul_mat           (ctx, model->output, t33);                  set_name(t34, "t34");     assert_shape_2d(t34, n_vocab, N*n_batch);
    struct ggml_tensor * t35   = ggml_reshape_3d        (ctx, t34, n_vocab, N, n_batch);            set_name(t35, "t35");     assert_shape_3d(t35, n_vocab, N, n_batch);
    struct ggml_tensor * t36   = ggml_cross_entropy_loss(ctx, t35, targets);                        set_name(t36, "t36");     assert_shape_1d(t36, 1);

    checkpoints.push_back(t31);
    checkpoints.push_back(t32);
    checkpoints.push_back(t33);
    checkpoints.push_back(t34);
    checkpoints.push_back(t35);
    checkpoints.push_back(t36);

    ggml_build_forward_expand(gf, t36);

    if (enable_checkpointing) {
        ggml_build_backward_gradient_checkpointing(ctx, gf, gb, gb_tmp, checkpoints.data(), (int) checkpoints.size());
    } else {
        *gb = *gf;
        ggml_build_backward_expand(ctx, gf, gb, true);
    }

    if (alloc) {
        // make sure some tensors are not reallocated by inserting new temporary nodes depending on them
        int n_leafs_before = gb->n_leafs;
        int n_nodes_before = gb->n_nodes;
        struct ggml_tensor * one = ggml_new_f32(ctx, 1.0f);
        // output tensors
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t35, one));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t36, one));
        // gradient tensors (will be set to zero by ggml_graph_reset)
        for (int i = 0; i < gf->n_nodes; ++i) {
            if (!gf->grads[i]) continue;
            if (gf->grads[i]->data == NULL && !ggml_is_view(gf->grads[i])) {
                ggml_allocr_alloc(alloc, gf->grads[i]);
            }
            // printf("%s:                g[%02d] data=[%p..%p] op=%s name=%s\n", __func__, i, gf->grads[i]->data, (char*) gf->grads[i]->data + ggml_nbytes(gf->grads[i]), ggml_op_name(gf->grads[i]->op), ggml_get_name(gf->grads[i]));
            ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, gf->grads[i], one));
        }
        for (int i = 0; i < checkpoints.size(); ++i) {
            if (checkpoints[i]->data == NULL && !ggml_is_view(checkpoints[i])) {
                ggml_allocr_alloc(alloc, checkpoints[i]);
            }
        }

        int n_leafs_after = gb->n_leafs;
        int n_nodes_after = gb->n_nodes;

        ggml_allocr_alloc_graph(alloc, gb);

        // remove the additional nodes and leafs
        for (int i = n_leafs_before; i < gb->n_leafs; ++i) {
            gb->leafs[i] = NULL;
        }
        for (int i = n_nodes_before; i < gb->n_nodes; ++i) {
            gb->nodes[i] = NULL;
        }
        gb->n_leafs = n_leafs_before;
        gb->n_nodes = n_nodes_before;
        // for (int i=0; i<gb->n_nodes; ++i) {
        //     struct ggml_tensor * node = gb->nodes[i];
        //     printf("%s: node[%d] data=[%p..%p]\n", __func__, i, node->data, (char*) node->data + ggml_nbytes(node));
        // }
    }

    *logits = t35;
    return t36;
}

void set_f32_3d(struct ggml_tensor * tensor, int64_t i0, int64_t i1, int64_t i2, float value) {
    float * ptr = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2]);
    *ptr = value;
}

void set_f32_2d(struct ggml_tensor * tensor, int64_t i0, int64_t i1, float value) {
    float * ptr = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
    *ptr = value;
}

void set_i32_2d(struct ggml_tensor * tensor, int64_t i0, int64_t i1, int32_t value) {
    int32_t * ptr = (int32_t *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
    *ptr = value;
}

float get_f32_2d(struct ggml_tensor * tensor, int64_t i0, int64_t i1) {
    float * ptr = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
    return *ptr;
}

int32_t get_i32_2d(struct ggml_tensor * tensor, int64_t i0, int64_t i1) {
    int32_t * ptr = (int32_t *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
    return *ptr;
}

void print_row(struct ggml_tensor * probs, int i) {
    for (int k = 0; k < probs->ne[0]; ++k) {
        float p = get_f32_2d(probs, k, i);
        printf(" %.2f", p);
    }
    printf("\n");
}

void print_matrix(struct ggml_tensor * probs) {
    assert(probs->n_dims == 2);
    for (int i = 0; i < probs->ne[1]; ++i) {
        for (int k = 0; k < probs->ne[0]; ++k) {
            float p = get_f32_2d(probs, k, i);
            printf(" %.2f", p);
        }
        printf("\n");
    }
}


void print_token(struct llama_context * ctx, llama_token token) {
    printf("%s", llama_token_to_str(ctx, token));
}

void print_tokens(struct llama_context* ctx, struct ggml_tensor * tokens) {
    for (int i=0; i<tokens->ne[0]; ++i) {
        int token = ggml_get_i32_1d(tokens, i);
        print_token(ctx, token);
    }
}

void print_tokens_batch(struct llama_context* ctx, struct ggml_tensor * tokens) {
    for (int i1=0; i1<tokens->ne[1]; ++i1) {
        //int num_newline = 0;
        for (int i0=0; i0<tokens->ne[0]; ++i0) {
            int token = get_i32_2d(tokens, i0, i1);
            print_token(ctx, token);
            // bool isnl = (token == llama_token_nl());
            // if (isnl) {
            //     ++num_newline;
            // }
            // if (isnl) {
            //     if (num_newline < 2) {
            //         print_token(ctx, token);
            //     } else {
            //         printf("\\n");
            //     }
            // } else {
            //     print_token(ctx, token);
            // }
        }
        printf("\n--\n");
    }
}

void get_example_targets(const int * train_samples, size_t n_train_samples, const llama_token * train_data, size_t n_train_data, int example_id, struct ggml_tensor * tokens_input, struct ggml_tensor * target_logits, struct ggml_tensor * target_probs) {
    int n_tokens = tokens_input->ne[0];
    int n_vocab  = target_logits->ne[0];

    size_t sample = train_samples[example_id % n_train_samples];
    GGML_ASSERT(sample+n_tokens-1 < n_train_data);

    ggml_set_f32(target_logits, -1.0f/n_vocab);
    ggml_set_f32(target_probs, 0.0f);
    ggml_set_i32_1d(tokens_input, 0, llama_token_bos());
    for (int i=1; i<n_tokens+1; ++i) {
        int token = clamp(train_data[sample+i-1], 0, n_vocab-1);
        set_f32_2d(target_logits, token, i-1, +1.0f);
        set_f32_2d(target_probs,  token, i-1, +1.0f);
        if (i<n_tokens) {
            ggml_set_i32_1d(tokens_input, i, token);
        }
    }
}

void get_example_targets_batch(const int * train_samples, size_t n_train_samples, const llama_token * train_data, size_t n_train_data, int example_id, struct ggml_tensor * tokens_input, struct ggml_tensor * target_logits, struct ggml_tensor * target_probs) {
    GGML_ASSERT(tokens_input->n_dims  == 2);
    GGML_ASSERT(target_logits->n_dims == 3);
    GGML_ASSERT(target_probs->n_dims  == 3);
    int n_vocab  = target_logits->ne[0];
    int n_tokens = tokens_input->ne[0];
    int n_batch  = tokens_input->ne[1];
    GGML_ASSERT(n_tokens == target_logits->ne[1]);
    GGML_ASSERT(n_batch  == target_logits->ne[2]);
    GGML_ASSERT(n_vocab  == target_probs->ne[0]);
    GGML_ASSERT(n_tokens == target_probs->ne[1]);
    GGML_ASSERT(n_batch  == target_probs->ne[2]);

    ggml_set_f32(target_logits, -1.0f/n_vocab);
    ggml_set_f32(target_probs, 0.0f);
    // printf("%s: example_id=%d n_batch=%d n_train_samples=%zu\n", __func__, example_id, n_batch, n_train_samples);
    for (int k=0; k<n_batch; ++k) {
        // printf("%s: batch %d\n", __func__, k);
        size_t sample_idx = (example_id*n_batch + k) % n_train_samples;
        size_t sample = train_samples[sample_idx];
        // printf("%s: sample_idx=%zu sample=%zu\n", __func__, sample_idx, sample);
        GGML_ASSERT(sample+n_tokens-1 < n_train_data);

        set_i32_2d(tokens_input, 0, k, llama_token_bos());
        for (int i=1; i<n_tokens+1; ++i) {
            int token = clamp(train_data[sample+i-1], 0, n_vocab-1);
            set_f32_3d(target_logits, token, i-1, k, +1.0f);
            set_f32_3d(target_probs,  token, i-1, k, +1.0f);
            if (i<n_tokens) {
                set_i32_2d(tokens_input, i, k, token);
            }
        }
    }
}


void lshift_examples(struct ggml_tensor * tokens_input, struct ggml_tensor * target_logits, struct ggml_tensor * target_probs, int n_shift) {
    int n_tokens = tokens_input->ne[0];
    int n_vocab = target_logits->ne[0];
    for (int i=0; i<n_tokens-n_shift; ++i) {
        ggml_set_i32_1d(tokens_input, i, ggml_get_i32_1d(tokens_input, i + n_shift));
        for (int k=0; k<n_vocab; ++k) {
            ggml_set_f32_1d(target_logits, i*n_vocab + k, ggml_get_f32_1d(target_logits, (i + n_shift)*n_vocab + k));
            ggml_set_f32_1d(target_probs, i*n_vocab + k,  ggml_get_f32_1d(target_probs,  (i + n_shift)*n_vocab + k));
        }
    }
}

struct ggml_tensor * square_error_loss(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * target) {
    return ggml_sum(ctx, ggml_sqr(ctx, ggml_sub(ctx, target, a)));
}

struct ggml_tensor * cross_entropy_loss(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * probs) {
    return ggml_cross_entropy_loss(ctx, a, probs);
}

#ifdef __GNUC__
#ifdef __MINGW32__
__attribute__((format(gnu_printf, 1, 2)))
#else
__attribute__((format(printf, 1, 2)))
#endif
#endif
static std::string format(const char * fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

struct llama_file {
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp;
    size_t size;

    llama_file(const char * fname, const char * mode) {
        fp = std::fopen(fname, mode);
        if (fp == NULL) {
            size = 0;
        } else {
            seek(0, SEEK_END);
            size = tell();
            seek(0, SEEK_SET);
        }
    }

    size_t tell() const {
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        GGML_ASSERT(ret != -1); // this really shouldn't fail
        return (size_t) ret;
    }

    void seek(size_t offset, int whence) {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        GGML_ASSERT(ret == 0); // same
    }

    void read_raw(void * ptr, size_t size) {
        if (size == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, size, 1, fp);
        if (ferror(fp)) {
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw std::runtime_error(std::string("unexpectedly reached end of file"));
        }
    }

    std::uint32_t read_u32() {
        std::uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    std::string read_string(std::uint32_t len) {
        std::vector<char> chars(len);
        read_raw(chars.data(), len);
        return std::string(chars.data(), len);
    }

    void write_raw(const void * ptr, size_t size) {
        if (size == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, size, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(std::uint32_t val) {
        write_raw(&val, sizeof(val));
    }

    ~llama_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
};

int tokenize_file(struct llama_context * lctx, const char * filename, std::vector<llama_token>& out) {
    struct llama_file f(filename, "rb");

    std::vector<char> buf;
    buf.resize(f.size+1);

    f.read_raw(buf.data(), f.size);
    buf[f.size] = '\0';

    out.resize(buf.size());

    int n_tokens = llama_tokenize(lctx, buf.data(), out.data(), buf.size(), false);
    if (n_tokens >= 0) {
        out.resize(n_tokens);
    }

    bool verify = false;
    if (verify) {
        const char * in  = buf.data();
        const char * end = buf.data() + buf.size();
        for (int i = 0; i < (int) out.size(); ++i) {
            const char * s = llama_token_to_str(lctx, out[i]);
            int len = strlen(s);
            if (in >= end) {
                printf("%s: unexpected end of original text.\n", __func__);
                break;
            }
            const bool matches = (strncmp(in, s, len) == 0);
            if (matches) {
                in += len;
            } else {
                printf("%s: mismatch: expected '%s', but got '%s'\n", __func__, std::string(in, len).c_str(), s);
            }
        }
    }

    return n_tokens;
}

void shuffle_ints(int * begin, int * end) {
    if (end <= begin) return;
    int max=begin[0];
    for (int i=1; i<end-begin; ++i) {
        if (begin[i] > max) {
            max = begin[i];
        }
    }
    std::vector<float> vals;
    vals.resize(max+1);
    for (int i=0; i<max+1; ++i) {
       vals[i] = frand();
    }
    std::sort(begin, end, [&vals](int a, int b){
       return vals.at(a) < vals.at(b);
    });
}

struct my_llama_sampler_params {
    float temp              = 0.0f;  // <= 0.0 disabled
    int   top_k             = 20;    // <= 0 to use vocab size
    float top_p             = 0.95f; // 1.0 = disabled
    float tfs_z             = 1.00f; // 1.0 = disabled
    float typical_p         = 1.00f; // 1.0 = disabled
    int   repeat_last_n     = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float repeat_penalty    = 1.0f;  // 1.0 = disabled
    float presence_penalty  = 0.0f;  // 0.0 = disabled
    float frequency_penalty = 0.0f;  // 0.0 = disabled
    int   mirostat          = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float mirostat_tau      = 5.00f; // target entropy
    float mirostat_eta      = 0.10f; // learning rate
    bool  penalize_nl       = true;  // consider newlines as a repeatable token
};

struct my_llama_sampler {
    struct llama_context * ctx = NULL;
    my_llama_sampler_params params;

    int n_vocab = 0;
    int n_ctx = 0;

    float mirostat_mu;

    std::vector<llama_token_data> candidates;
    llama_token_data_array candidates_p;

};

void init_sampler(struct my_llama_sampler * sampler, struct llama_context * ctx) {
    sampler->ctx = ctx;
    sampler->n_vocab = llama_n_vocab(sampler->ctx);
    sampler->n_ctx   = llama_n_ctx(sampler->ctx);
    sampler->mirostat_mu = 2.0f * sampler->params.mirostat_tau;
}

llama_token sample(struct my_llama_sampler * sampler, float * logits, const llama_token * last_tokens, int n_last_tokens) {
    GGML_ASSERT(sampler->ctx != NULL);

    struct llama_context * ctx = sampler->ctx;

    sampler->candidates.resize(sampler->n_vocab);
    for (llama_token token_id = 0; token_id < sampler->n_vocab; ++token_id) {
        sampler->candidates[token_id].id = token_id;
        sampler->candidates[token_id].logit = logits[token_id];
        sampler->candidates[token_id].p = 0.0;
    }

    llama_token_data_array * candidates_p = & sampler->candidates_p;

    candidates_p->data = sampler->candidates.data();
    candidates_p->size = sampler->candidates.size();
    candidates_p->sorted = false;

    const auto params = sampler->params;

    // Apply penalties
    const float nl_logit = logits[llama_token_nl()];

    const int n_last = std::min(std::min(n_last_tokens, params.repeat_last_n), sampler->n_ctx);

    llama_sample_repetition_penalty(
        ctx,
        candidates_p,
        last_tokens + n_last_tokens - n_last,
        n_last,
        params.repeat_penalty);
    llama_sample_frequency_and_presence_penalties(
        ctx,
        candidates_p,
        last_tokens + n_last_tokens - n_last,
        n_last,
        params.frequency_penalty,
        params.presence_penalty);

    if (!params.penalize_nl) {
        logits[llama_token_nl()] = nl_logit;
    }

    llama_token token = 0;
    if (params.temp <= 0) {
        // Greedy sampling
        token = llama_sample_token_greedy(ctx, candidates_p);
    } else {
        if (params.mirostat == 1) {
            int mirostat_m = 100;
            llama_sample_temperature(ctx, candidates_p, params.temp);
            token = llama_sample_token_mirostat(ctx, candidates_p, params.mirostat_tau, params.mirostat_eta, mirostat_m, &sampler->mirostat_mu);
        } else if (params.mirostat == 2) {
            llama_sample_temperature(ctx, candidates_p, params.temp);
            token = llama_sample_token_mirostat_v2(ctx, candidates_p, params.mirostat_tau, params.mirostat_eta, &sampler->mirostat_mu);
        } else {
            // Temperature sampling
            llama_sample_top_k        (ctx, candidates_p, params.top_k, 1);
            llama_sample_tail_free    (ctx, candidates_p, params.tfs_z, 1);
            llama_sample_typical      (ctx, candidates_p, params.typical_p, 1);

            llama_sample_top_p        (ctx, candidates_p, params.top_p, 1);
            llama_sample_temperature  (ctx, candidates_p, params.temp);
            token = llama_sample_token(ctx, candidates_p);
        }
    }
    return token;
}

void set_logits_masked(struct ggml_tensor * logits, std::vector<bool>& mask, float value) {
    GGML_ASSERT(logits->ne[0] == (int64_t) mask.size());
    for (int i2 = 0; i2 < logits->ne[2]; ++i2) {
        for (int i1 = 0; i1 < logits->ne[1]; ++i1) {
            for (int i0 = 0; i0 < logits->ne[0]; ++i0) {
                if (!mask[i0]) continue;
                float * ptr = (float *) ((char *) logits->data + i2*logits->nb[2] + i1*logits->nb[1] + i0*logits->nb[0]);
                *ptr = value;
            }
        }
    }
}

void write_tensor(struct llama_file * file, struct ggml_tensor * tensor) {
    if (tensor == NULL) {
        file->write_u32(0);
        file->write_u32(0);
        file->write_u32(GGML_TYPE_F32);
        file->seek((0-file->tell()) & 31, SEEK_CUR);
        return;
    }
    const char * name = ggml_get_name(tensor);
    uint32_t name_len = strlen(name);
    uint32_t nd = tensor->n_dims;
    uint32_t ne[4] = { (uint32_t)tensor->ne[0],
                       (uint32_t)tensor->ne[1],
                       (uint32_t)tensor->ne[2],
                       (uint32_t)tensor->ne[3] };
    file->write_u32(nd);
    file->write_u32(name_len);
    file->write_u32(tensor->type);
    file->write_raw(ne, sizeof(ne[0]) * nd);
    file->write_raw(name, name_len);
    file->seek((0-file->tell()) & 31, SEEK_CUR);
    file->write_raw(tensor->data, ggml_nbytes(tensor));
}

void read_tensor(struct llama_file * file, struct ggml_tensor * tensor) {
    int32_t nd = file->read_u32();
    GGML_ASSERT(nd == tensor->n_dims);

    uint32_t name_len       = file->read_u32();
    enum     ggml_type type = (enum ggml_type) file->read_u32();
    GGML_ASSERT(type == tensor->type);

    uint32_t ne[4];
    file->read_raw(ne, sizeof(ne[0]) * nd);
    for (int i=0; i<nd; ++i) {
        GGML_ASSERT(ne[i] == tensor->ne[i]);
    }

    std::string name = file->read_string(name_len);
    GGML_ASSERT(strncmp(ggml_get_name(tensor), name.c_str(), sizeof(tensor->name)-1) == 0);

    file->seek((0-file->tell()) & 31, SEEK_CUR);
    file->read_raw(tensor->data, ggml_nbytes(tensor));
}

void skip_tensor(struct llama_file * file) {
    int32_t nd = file->read_u32();

    uint32_t name_len       = file->read_u32();
    enum     ggml_type type = (enum ggml_type) file->read_u32();

    uint32_t ne[4] = { 1, 1, 1, 1 };

    file->read_raw(ne, sizeof(ne[0]) * nd);

    std::string name = file->read_string(name_len);

    file->seek(-file->tell() & 31, SEEK_CUR);

    size_t nelements = ne[0]*ne[1]*ne[2]*ne[3];
    size_t nbytes = nelements*ggml_type_size(type)/ggml_blck_size(type);
    file->seek(nbytes, SEEK_CUR);
}

void write_opt_context(struct llama_file * file, struct ggml_opt_context * opt) {
    const uint32_t version = 1;
    GGML_ASSERT(opt->nx   >= 0);
    GGML_ASSERT(opt->iter >= 0);
    file->write_u32(version);
    file->write_u32(opt->params.past);
    file->write_u32(opt->params.lbfgs.m);
    file->write_raw(&opt->nx,     sizeof(opt->nx));
    file->write_raw(&opt->iter,   sizeof(opt->iter));
    file->write_u32((uint32_t)  opt->just_initialized);
    switch (opt->params.type) {
        case GGML_OPT_ADAM:
            {
                GGML_ASSERT(opt->adam.m  != NULL);
                GGML_ASSERT(opt->adam.v  != NULL);
                write_tensor(file, opt->adam.m);
                write_tensor(file, opt->adam.v);
                write_tensor(file, opt->adam.pf);
                file->write_raw(&opt->adam.fx_best,          sizeof(opt->adam.fx_best));
                file->write_raw(&opt->adam.fx_prev,          sizeof(opt->adam.fx_prev));
                file->write_raw(&opt->adam.n_no_improvement, sizeof(opt->adam.n_no_improvement));
            } break;
        case GGML_OPT_LBFGS:
            {
                GGML_ASSERT(opt->lbfgs.x != NULL);
                write_tensor(file, opt->lbfgs.x);
                write_tensor(file, opt->lbfgs.xp);
                write_tensor(file, opt->lbfgs.g);
                write_tensor(file, opt->lbfgs.gp);
                write_tensor(file, opt->lbfgs.d);
                write_tensor(file, opt->lbfgs.pf);
                write_tensor(file, opt->lbfgs.lmal);
                write_tensor(file, opt->lbfgs.lmys);
                write_tensor(file, opt->lbfgs.lms);
                write_tensor(file, opt->lbfgs.lmy);
                file->write_raw(&opt->lbfgs.fx_best,          sizeof(opt->lbfgs.fx_best));
                file->write_raw(&opt->lbfgs.step,             sizeof(opt->lbfgs.step));
                file->write_raw(&opt->lbfgs.j,                sizeof(opt->lbfgs.j));
                file->write_raw(&opt->lbfgs.k,                sizeof(opt->lbfgs.k));
                file->write_raw(&opt->lbfgs.end,              sizeof(opt->lbfgs.end));
                file->write_raw(&opt->lbfgs.n_no_improvement, sizeof(opt->lbfgs.n_no_improvement));
            } break;
    }
}

struct ggml_opt_params_v0 {
    enum ggml_opt_type type;
    int n_threads;
    int past;
    float delta;
    int max_no_improvement;
    bool print_forward_graph;
    bool print_backward_graph;
    struct {
        int n_iter;
        float sched;
        float decay;
        float alpha;
        float beta1;
        float beta2;
        float eps;
        float eps_f;
        float eps_g;
    } adam;
    struct {
        int m;
        int n_iter;
        int max_linesearch;
        float eps;
        float ftol;
        float wolfe;
        float min_step;
        float max_step;
        enum ggml_linesearch linesearch;
    } lbfgs;
};

void read_opt_context_v0(struct llama_file * file, struct ggml_context * ctx, struct ggml_opt_context * opt) {
    ggml_opt_params_v0 pv0;
    file->read_raw(&pv0, sizeof(pv0));
    opt->params.past = pv0.past;
    opt->params.lbfgs.m = pv0.lbfgs.m;
    file->read_raw(&opt->nx, sizeof(opt->nx));
    ggml_opt_init(ctx, opt, opt->params, opt->nx);

    file->read_raw(&opt->iter,   sizeof(opt->iter));
    opt->just_initialized = (bool) file->read_u32();

    switch (opt->params.type) {
        case GGML_OPT_ADAM:
            {
                skip_tensor(file);
                skip_tensor(file);
                skip_tensor(file);
                read_tensor(file, opt->adam.m);
                read_tensor(file, opt->adam.v);
                skip_tensor(file);
                skip_tensor(file);
                if (opt->adam.pf) { read_tensor(file, opt->adam.pf); }
                file->read_raw(&opt->adam.fx_best,          sizeof(opt->adam.fx_best));
                file->read_raw(&opt->adam.fx_prev,          sizeof(opt->adam.fx_prev));
                file->read_raw(&opt->adam.n_no_improvement, sizeof(opt->adam.n_no_improvement));
            } break;
        case GGML_OPT_LBFGS:
            {
                GGML_ASSERT(opt->lbfgs.x != NULL);
                read_tensor(file, opt->lbfgs.x);
                read_tensor(file, opt->lbfgs.xp);
                read_tensor(file, opt->lbfgs.g);
                read_tensor(file, opt->lbfgs.gp);
                read_tensor(file, opt->lbfgs.d);
                if (opt->lbfgs.pf) { read_tensor(file, opt->lbfgs.pf); }
                read_tensor(file, opt->lbfgs.lmal);
                read_tensor(file, opt->lbfgs.lmys);
                read_tensor(file, opt->lbfgs.lms);
                read_tensor(file, opt->lbfgs.lmy);
                file->read_raw(&opt->lbfgs.fx_best,          sizeof(opt->lbfgs.fx_best));
                file->read_raw(&opt->lbfgs.step,             sizeof(opt->lbfgs.step));
                file->read_raw(&opt->lbfgs.j,                sizeof(opt->lbfgs.j));
                file->read_raw(&opt->lbfgs.k,                sizeof(opt->lbfgs.k));
                file->read_raw(&opt->lbfgs.end,              sizeof(opt->lbfgs.end));
                file->read_raw(&opt->lbfgs.n_no_improvement, sizeof(opt->lbfgs.n_no_improvement));
            } break;
    }
}

void read_opt_context_v1(struct llama_file * file, struct ggml_context * ctx, struct ggml_opt_context * opt) {
    opt->params.past    = (int) file->read_u32();
    opt->params.lbfgs.m = (int) file->read_u32();
    file->read_raw(&opt->nx,     sizeof(opt->nx));
    ggml_opt_init(ctx, opt, opt->params, opt->nx);

    file->read_raw(&opt->iter,   sizeof(opt->iter));
    opt->just_initialized = (bool) file->read_u32();

    switch (opt->params.type) {
        case GGML_OPT_ADAM:
            {
                read_tensor(file, opt->adam.m);
                read_tensor(file, opt->adam.v);
                if (opt->adam.pf) { read_tensor(file, opt->adam.pf); }
                file->read_raw(&opt->adam.fx_best,          sizeof(opt->adam.fx_best));
                file->read_raw(&opt->adam.fx_prev,          sizeof(opt->adam.fx_prev));
                file->read_raw(&opt->adam.n_no_improvement, sizeof(opt->adam.n_no_improvement));
            } break;
        case GGML_OPT_LBFGS:
            {
                GGML_ASSERT(opt->lbfgs.x != NULL);
                read_tensor(file, opt->lbfgs.x);
                read_tensor(file, opt->lbfgs.xp);
                read_tensor(file, opt->lbfgs.g);
                read_tensor(file, opt->lbfgs.gp);
                read_tensor(file, opt->lbfgs.d);
                if (opt->lbfgs.pf) { read_tensor(file, opt->lbfgs.pf); }
                read_tensor(file, opt->lbfgs.lmal);
                read_tensor(file, opt->lbfgs.lmys);
                read_tensor(file, opt->lbfgs.lms);
                read_tensor(file, opt->lbfgs.lmy);
                file->read_raw(&opt->lbfgs.fx_best,          sizeof(opt->lbfgs.fx_best));
                file->read_raw(&opt->lbfgs.step,             sizeof(opt->lbfgs.step));
                file->read_raw(&opt->lbfgs.j,                sizeof(opt->lbfgs.j));
                file->read_raw(&opt->lbfgs.k,                sizeof(opt->lbfgs.k));
                file->read_raw(&opt->lbfgs.end,              sizeof(opt->lbfgs.end));
                file->read_raw(&opt->lbfgs.n_no_improvement, sizeof(opt->lbfgs.n_no_improvement));
            } break;
    }
}

void read_opt_context(struct llama_file * file, struct ggml_context * ctx, struct ggml_opt_context * opt) {
    uint32_t version = file->read_u32();
    printf("%s: opt context version %u\n", __func__, version);
    switch (version) {
        case 0:
            {
                read_opt_context_v0(file, ctx, opt);
            } break;
        case 1:
            {
                read_opt_context_v1(file, ctx, opt);
            } break;
        default:
            {
                fprintf(stderr, "%s: unknown version %u\n", __func__, version);
            }
    }
}

void save_checkpoint(struct my_llama_model * model, struct ggml_opt_context * opt, const char * filename) {
    struct llama_file file(filename, "wb");
    if (file.fp == NULL) {
        return;
    }

    const uint32_t magic   = 'ggcp';
    const uint32_t version = 0;

    file.write_u32(magic);
    file.write_u32(version);
    file.write_u32(model->train_its);
    file.write_u32(model->train_samples);
    file.write_u32(model->train_tokens);
    file.write_u32(model->hparams.n_vocab);
    file.write_u32(model->hparams.n_embd);
    file.write_u32(model->hparams.n_mult);
    file.write_u32(model->hparams.n_head);
    file.write_u32(model->hparams.n_layer);
    file.write_u32(model->hparams.n_rot);

    write_tensor(&file, model->tok_embeddings);
    write_tensor(&file, model->norm);
    write_tensor(&file, model->output);

    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
        auto & layer = model->layers[i];

        write_tensor(&file, layer.attention_norm);
        write_tensor(&file, layer.wq);
        write_tensor(&file, layer.wk);
        write_tensor(&file, layer.wv);
        write_tensor(&file, layer.wo);
        write_tensor(&file, layer.ffn_norm);
        write_tensor(&file, layer.w1);
        write_tensor(&file, layer.w2);
        write_tensor(&file, layer.w3);
    }

    write_opt_context(&file, opt);
}

bool load_checkpoint(struct my_llama_model * model, struct ggml_opt_context * opt, const char * filename, bool init) {
    struct llama_file file(filename, "rb");

    uint32_t magic;
    uint32_t version;

    uint32_t train_its = 0;
    uint32_t train_samples = 0;
    uint32_t train_tokens = 0;

    if (file.fp) {
        printf("%s: Loading model from '%s'.\n", __func__, filename);
        magic                  = file.read_u32();
        GGML_ASSERT(magic     == 'ggcp');
        version                = file.read_u32();
        GGML_ASSERT(version   == 0);
        train_its              = file.read_u32();
        train_samples          = file.read_u32();
        train_tokens           = file.read_u32();
        model->hparams.n_vocab = file.read_u32();
        model->hparams.n_embd  = file.read_u32();
        model->hparams.n_mult  = file.read_u32();
        model->hparams.n_head  = file.read_u32();
        model->hparams.n_layer = file.read_u32();
        model->hparams.n_rot   = file.read_u32();
        print_params(&model->hparams);
    }

    if (init) {
        init_model(model);
    }

    if (file.fp) {
        model->train_its = train_its;
        model->train_samples = train_samples;
        model->train_tokens = train_tokens;
    }

    printf("%s: Training iterations: %u.\n", __func__, model->train_its);
    printf("%s: Training samples:    %u.\n", __func__, model->train_samples);
    printf("%s: Training tokens:     %u.\n", __func__, model->train_tokens);

    if (file.fp) {
        read_tensor(&file, model->tok_embeddings);
        read_tensor(&file, model->norm);
        read_tensor(&file, model->output);

        for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
            auto & layer = model->layers[i];

            read_tensor(&file, layer.attention_norm);
            read_tensor(&file, layer.wq);
            read_tensor(&file, layer.wk);
            read_tensor(&file, layer.wv);
            read_tensor(&file, layer.wo);
            read_tensor(&file, layer.ffn_norm);
            read_tensor(&file, layer.w1);
            read_tensor(&file, layer.w2);
            read_tensor(&file, layer.w3);
        }

        read_opt_context(&file, model->ctx, opt);
    }

    return (file.fp != NULL);
}

void save_as_llama_model(struct llama_vocab * vocab, struct my_llama_model * model, const char * filename) {
    struct llama_file file(filename, "wb");
    if (file.fp == NULL) {
        return;
    }

    // write_magic
    file.write_u32(LLAMA_FILE_MAGIC);   // magic
    file.write_u32(LLAMA_FILE_VERSION); // version
    // write_hparams
    file.write_u32(model->hparams.n_vocab);
    file.write_u32(model->hparams.n_embd);
    file.write_u32(model->hparams.n_mult);
    file.write_u32(model->hparams.n_head);
    file.write_u32(model->hparams.n_layer);
    file.write_u32(model->hparams.n_rot);
    file.write_u32(LLAMA_FTYPE_ALL_F32);
    // write_vocab
    uint32_t n_vocab = model->hparams.n_vocab;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const auto & token_score = vocab->id_to_token.at(i);
        file.write_u32((uint32_t) token_score.tok.size());
        file.write_raw(token_score.tok.data(), token_score.tok.size());
        file.write_raw(&token_score.score, sizeof(token_score.score));
    }
    // write tensors
    write_tensor(&file, model->tok_embeddings);
    write_tensor(&file, model->norm);
    write_tensor(&file, model->output);
    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
        auto & layer = model->layers[i];

        write_tensor(&file, layer.attention_norm);
        write_tensor(&file, layer.wq);
        write_tensor(&file, layer.wk);
        write_tensor(&file, layer.wv);
        write_tensor(&file, layer.wo);
        write_tensor(&file, layer.ffn_norm);
        write_tensor(&file, layer.w1);
        write_tensor(&file, layer.w2);
        write_tensor(&file, layer.w3);
    }
}

float cosine_decay(const int decay_steps, const float minimum, int step) {
    if (step > decay_steps) {
        step = decay_steps;
    }
    const float cosine_decay = 0.50f*(1.0f + cosf(3.14159265359f*step/decay_steps));
    const float decay = (1 - minimum)*cosine_decay + minimum;
    return decay;
}

float cosine_decay_restart(int decay_steps, const float minimum, int step, float restart_step_mult, bool enable_restart) {
    if (enable_restart) {
        while (step > decay_steps) {
            step -= decay_steps;
            decay_steps = (int) restart_step_mult * decay_steps;
        }
    }
    return cosine_decay(decay_steps, minimum, step);
}

struct train_params {
    const char * fn_vocab_model;
    const char * fn_train_data;
    const char * fn_checkpoint_in;
    const char * fn_checkpoint_out;
    const char * fn_model_out;

    uint32_t seed;

    int n_ctx;
    int n_embd;
    int n_mult;
    int n_head;
    int n_layer;
    int n_rotmax;

    int n_threads;
    int n_batch;
    int n_examples;
    int n_predict;

    int print_info_interval;
    int print_details_interval;

    bool samples_start_after_nl;
    bool use_adam;
    bool use_flash;
    bool use_checkpointing;
    bool use_alloc;

    // only adam
    int   warmup;
    int   cos_decay_steps;
    float cos_decay_restart;
    float cos_decay_min;
    bool  enable_restart;

    int   opt_past;
    float opt_delta;
    int   opt_max_no_improvement;

    int   lbfgs_n_iter;
    int   adam_n_iter;
    float adam_alpha;
    float adam_min_alpha;
    float adam_decay;
    int   adam_decay_min_ndim;
    float adam_beta1;
    float adam_beta2;
    float adam_gclip;
    float adam_eps_f;

    int mem_model_gb;
    int mem_compute_gb;
    int mem_compute0_gb;
};

struct train_params get_default_train_params() {
    struct train_params params;
    params.fn_vocab_model    = "ggml-vic7b-uncensored-q4_0.bin";
    params.fn_train_data     = "shakespeare.txt";
    params.fn_checkpoint_in  = "checkpoint.bin";
    params.fn_checkpoint_out = "checkpoint.bin";
    params.fn_model_out      = "ggml-checkpoint-f32.bin";

    params.seed       =   -1;

    params.n_ctx      =  128;
    params.n_embd     =  256;
    params.n_mult     =  256;
    params.n_head     =    8;
    params.n_layer    =   16;
    params.n_rotmax   =   64;

    params.n_threads  =    6;
    params.n_batch    =    8;
    params.n_examples =    1;
    params.n_predict  = 1024;

    params.print_info_interval    = 1;
    params.print_details_interval = 2;

    params.samples_start_after_nl = false;
    params.use_adam               = true;
    params.use_flash              = true;
    params.use_checkpointing      = true;
    params.use_alloc              = true;

    params.opt_past               = 0;
    params.opt_delta              = 1e-5f;
    params.opt_max_no_improvement = 0;

    // only adam
    params.warmup            =  100;
    params.cos_decay_steps   = 1000;
    params.cos_decay_restart = 1.1f;
    params.cos_decay_min     = 0.1f;
    params.enable_restart    = false;

    params.lbfgs_n_iter        = 256;
    params.adam_n_iter         = 256;
    params.adam_alpha          = 1e-3f;
    params.adam_min_alpha      = 0;
    params.adam_decay          = 1e-1f;
    params.adam_decay_min_ndim = 2;
    params.adam_beta1          = 0.9f;
    params.adam_beta2          = 0.999f;
    params.adam_gclip          = 1.0f;
    params.adam_eps_f          = 0.0f;

    params.mem_model_gb   =  2;
    params.mem_compute_gb = 24;
    params.mem_compute0_gb = 8;
    return params;
}

void train_print_usage(int /*argc*/, char ** argv, const struct train_params * params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                 show this help message and exit\n");
    fprintf(stderr, "  --vocab-model FNAME        model path from which to load vocab (default '%s')\n", params->fn_vocab_model);
    fprintf(stderr, "  --train-data FNAME         path from which to load training data (default '%s')\n", params->fn_train_data);
    fprintf(stderr, "  --checkpoint-in FNAME      path from which to load training checkpoint (default '%s')\n", params->fn_checkpoint_in);
    fprintf(stderr, "  --checkpoint-out FNAME     path to save training checkpoint (default '%s')\n", params->fn_checkpoint_out);
    fprintf(stderr, "  --model-out FNAME          path to save ggml model (default '%s')\n", params->fn_model_out);
    fprintf(stderr, "  -s SEED, --seed SEED       RNG seed (default: -1, use random seed for -1)\n");
    fprintf(stderr, "  -c N, --ctx N              Context size used during training (default %d)\n", params->n_ctx);
    fprintf(stderr, "  --embd N                   Embedding size used for new models (default %d)\n", params->n_embd);
    fprintf(stderr, "  --mult N                   Mult size used for new models, influences feedforward size. (default %d)\n", params->n_mult);
    fprintf(stderr, "  --head N                   Number of heads for new models (default %d)\n", params->n_head);
    fprintf(stderr, "  --layer N                  Number of layers for new models (default %d)\n", params->n_layer);
    fprintf(stderr, "  --rotmax N                 Maximal number Rope dimensions for new models (default %d)\n", params->n_rotmax);
    fprintf(stderr, "  -t N, --threads N          Number of threads (default %d)\n", params->n_threads);
    fprintf(stderr, "  -b N, --batch N            Parallel batch size (default %d)\n", params->n_batch);
    fprintf(stderr, "  -n N, --examples N         Number of examples to train (default %d)\n", params->n_examples);
    fprintf(stderr, "  --predict N                Number of tokens to generate after training (default %d)\n", params->n_predict);
    fprintf(stderr, "  --print-info-interval N    Print infos during training each N examples (default %d)\n", params->print_info_interval);
    fprintf(stderr, "  --print-details-interval N Print details during training each N examples (default %d)\n", params->print_details_interval);
    fprintf(stderr, "  --samples-after-nl         Training samples start after newlines. (default %s)\n", params->samples_start_after_nl ? "on" : "off");
    fprintf(stderr, "  --use-lbfgs                Use LBFGS optimizer instead of default Adam\n");
    fprintf(stderr, "  --use-adam                 Use Adam optimizer (default)\n");
    fprintf(stderr, "  --no-flash                 Don't use flash attention \n");
    fprintf(stderr, "  --use-flash                Use flash attention (default)\n");
    fprintf(stderr, "  --no-checkpointing         Don't use gradient checkpointing\n");
    fprintf(stderr, "  --use-checkpointing        Use gradient checkpointing (default)\n");
    fprintf(stderr, "  --no-alloc                 Don't use allocator\n");
    fprintf(stderr, "  --use-alloc                Use allocator (default)\n");
    fprintf(stderr, "  --warmup N                 Only for Adam optimizer. Number of warmup steps (default %d)\n", params->warmup);
    fprintf(stderr, "  --cos-decay-steps N        Only for Adam optimizer. Number of cosine decay steps (default %d)\n", params->cos_decay_steps);
    fprintf(stderr, "  --cos-decay-restart N      Only for Adam optimizer. Increase of cosine decay steps after restart (default %f)\n", params->cos_decay_restart);
    fprintf(stderr, "  --cos-decay-min N          Only for Adam optimizer. Cosine decay minimum (default %f)\n", params->cos_decay_min);
    fprintf(stderr, "  --enable-restart N         Only for Adam optimizer. Enable restarts of cos-decay %s\n", params->enable_restart ? "(default)" : "");
    fprintf(stderr, "  --disable-restart N        Only for Adam optimizer. Disable restarts of cos-decay %s\n", !params->enable_restart ? "(default)" : "");
    fprintf(stderr, "  --opt-past N               Number of optimization iterations to track for delta convergence test. Disabled when zero. (default %d)\n", params->opt_past);
    fprintf(stderr, "  --opt-delta N              Maximum delta for delta convergence test. Disabled when <= zero. (default %f)\n", params->opt_delta);
    fprintf(stderr, "  --opt-max-no-improvement N Maximum number of optimization iterations with no improvement. Disabled when <= zero. (default %d)\n", params->opt_max_no_improvement);
    fprintf(stderr, "  --adam-epsf N              AdamW epsilon for convergence test. Disabled when <= zero. (default %f)\n", params->adam_eps_f);
    fprintf(stderr, "  --adam-iter N              Maximum number of Adam optimization iterations for each batch (default %d)\n", params->adam_n_iter);
    fprintf(stderr, "  --adam-alpha N             Adam learning rate alpha (default %f)\n", params->adam_alpha);
    fprintf(stderr, "  --adam-min-alpha N         Adam minimum learning rate alpha - including warmup phase (default %f)\n", params->adam_min_alpha);
    fprintf(stderr, "  --adam-decay N             AdamW weight decay. Values greater zero enable AdamW instead of regular Adam. (default %f)\n", params->adam_decay);
    fprintf(stderr, "  --adam-decay-min-ndim N    Minimum number of tensor dimensions to apply AdamW weight decay. Weight decay is not applied to tensors with less n_dims. (default %d)\n", params->adam_decay_min_ndim);
    fprintf(stderr, "  --adam-beta1 N             AdamW beta1 in interval [0,1). How much to smooth the first moment of gradients. (default %f)\n", params->adam_beta1);
    fprintf(stderr, "  --adam-beta2 N             AdamW beta2 in interval [0,1). How much to smooth the second moment of gradients. (default %f)\n", params->adam_beta2);
    fprintf(stderr, "  --adam-gclip N             AdamW gradient clipping. Disabled when zero. (default %f)\n", params->adam_gclip);
    fprintf(stderr, "  --lbfgs-iter N             Maximum number of LBFGS optimization iterations for each batch (default %d)\n", params->lbfgs_n_iter);
    fprintf(stderr, "  --mem-model N              Memory to allocate for model and cache in gigabytes. (default %d)\n", params->mem_model_gb);
    fprintf(stderr, "  --mem-compute N            Memory to allocate for compute in gigabytes. (default %d)\n", params->mem_compute_gb);
    fprintf(stderr, "  --mem-compute0 N           Memory to allocate for automatic memory allocator in gigabytes. (default %d)\n", params->mem_compute0_gb);
    fprintf(stderr, "\n");
}

bool train_params_parse(int argc, char ** argv, struct train_params * params) {
    bool invalid_param = false;
    std::string arg;
    struct train_params default_params = get_default_train_params();
    const std::string arg_prefix = "--";

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "--vocab-model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_vocab_model = argv[i];
        } else if (arg == "--train-data") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_train_data = argv[i];
        } else if (arg == "--checkpoint-in") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_checkpoint_in = argv[i];
        } else if (arg == "--checkpoint-out") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_checkpoint_out = argv[i];
        } else if (arg == "--model-out") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_model_out = argv[i];
        } else if (arg == "-s" || arg == "--seed") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->seed = std::stoi(argv[i]);
        } else if (arg == "-c" || arg == "--ctx") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_ctx = std::stoi(argv[i]);
        } else if (arg == "--embd") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_embd = std::stoi(argv[i]);
        } else if (arg == "--mult") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_mult = std::stoi(argv[i]);
        } else if (arg == "--head") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_head = std::stoi(argv[i]);
        } else if (arg == "--layer") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_layer = std::stoi(argv[i]);
        } else if (arg == "--rotmax") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_rotmax = std::stoi(argv[i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_threads = std::stoi(argv[i]);
        } else if (arg == "-b" || arg == "--batch") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_batch = std::stoi(argv[i]);
        } else if (arg == "-n" || arg == "--examples") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_examples = std::stoi(argv[i]);
        } else if (arg == "--predict") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_predict = std::stoi(argv[i]);
        } else if (arg == "--print-info-interval") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->print_info_interval = std::stoi(argv[i]);
        } else if (arg == "--print-details-interval") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->print_details_interval = std::stoi(argv[i]);
        } else if (arg == "--samples-after-nl") {
            params->samples_start_after_nl = true;
        } else if (arg == "--use-lbfgs") {
            params->use_adam = false;
        } else if (arg == "--use-adam") {
            params->use_adam = true;
        } else if (arg == "--no-flash") {
            params->use_flash = false;
        } else if (arg == "--use-flash") {
            params->use_flash = true;
        } else if (arg == "--no-checkpointing") {
            params->use_checkpointing = false;
        } else if (arg == "--use-checkpointing") {
            params->use_checkpointing = true;
        } else if (arg == "--no-alloc") {
            params->use_alloc = false;
        } else if (arg == "--use-alloc") {
            params->use_alloc = true;
        } else if (arg == "--warmup") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->warmup = std::stoi(argv[i]);
        } else if (arg == "--cos-decay-steps") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->cos_decay_steps = std::stof(argv[i]);
        } else if (arg == "--cos-decay-restart") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->cos_decay_restart = std::stof(argv[i]);
        } else if (arg == "--cos-decay-min") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->cos_decay_min = std::stof(argv[i]);
        } else if (arg == "--enable-restart") {
            params->enable_restart = true;
        } else if (arg == "--disable-restart") {
            params->enable_restart = false;
        } else if (arg == "--opt-past") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->opt_past = std::stoi(argv[i]);
        } else if (arg == "--opt-delta") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->opt_delta = std::stof(argv[i]);
        } else if (arg == "--opt-max-no-improvement") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->opt_max_no_improvement = std::stoi(argv[i]);
        } else if (arg == "--adam-epsf") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_eps_f = std::stof(argv[i]);
        } else if (arg == "--adam-iter") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_n_iter = std::stoi(argv[i]);
        } else if (arg == "--adam-alpha") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_alpha = std::stof(argv[i]);
        } else if (arg == "--adam-min-alpha") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_min_alpha = std::stof(argv[i]);
        } else if (arg == "--adam-decay") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_decay = std::stof(argv[i]);
        } else if (arg == "--adam-decay-min-ndim") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_decay_min_ndim = std::stoi(argv[i]);
        } else if (arg == "--adam-beta1") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_beta1 = std::stof(argv[i]);
        } else if (arg == "--adam-beta2") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_beta2 = std::stof(argv[i]);
        } else if (arg == "--adam-gclip") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_gclip = std::stof(argv[i]);
        } else if (arg == "--lbfgs-iter") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->lbfgs_n_iter = std::stoi(argv[i]);
        } else if (arg == "--mem-model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->mem_model_gb = std::stoi(argv[i]);
        } else if (arg == "--mem-compute") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->mem_compute_gb = std::stoi(argv[i]);
        } else if (arg == "--mem-compute0") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->mem_compute0_gb = std::stoi(argv[i]);
        } else if (arg == "-h" || arg == "--help") {
            train_print_usage(argc, argv, &default_params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            train_print_usage(argc, argv, &default_params);
            exit(1);
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        train_print_usage(argc, argv, &default_params);
        exit(1);
    }

    return true;
}

struct opt_callback_data {
    struct train_params *     params;
    struct ggml_opt_context * opt;
    llama_token *             tokens_data;
    size_t                    tokens_size;
    int *                     samples_data;
    size_t                    samples_size;
    int                       shuffle_countdown;
    struct ggml_tensor *      tokens_input;
    struct ggml_tensor *      target_logits;
    struct ggml_tensor *      target_probs;
};

void opt_callback(void * vdata, float * sched) {
    struct opt_callback_data * data = (struct opt_callback_data *) vdata;
    struct train_params * params    = data->params;
    struct ggml_opt_context * opt   = data->opt;
    int n_batch = params->n_batch;

    *sched = (opt->iter < params->warmup)
                ? (float) opt->iter / (float) params->warmup
                : cosine_decay_restart(
                    params->cos_decay_steps,
                    params->cos_decay_min,
                    opt->iter - params->warmup,
                    params->cos_decay_restart,
                    params->enable_restart);
    float min_sched = params->adam_min_alpha / params->adam_alpha;
    *sched = min_sched + *sched * (1.0f - min_sched);

    int impr_plot = -(int)(1 + (opt->loss_before - opt->loss_after) * 10.0f + 0.5f);
    printf("%s: iter=%*d, sched=%f loss0=%f loss=%f | improvement: %*d>\n", __func__, 6, opt->iter, *sched, opt->loss_before, opt->loss_after, impr_plot, (int)0);

    if (data->shuffle_countdown < n_batch) {
        printf("%s: reshuffle samples\n", __func__);
        shuffle_ints(data->samples_data, data->samples_data + data->samples_size);
        for (int i = 0; i < (int) data->samples_size; ++i) {
            GGML_ASSERT(data->samples_data[i]+params->n_ctx-1 < (int) data->tokens_size);
        }
        data->shuffle_countdown = data->samples_size;
    }

    get_example_targets_batch(
        data->samples_data,
        data->samples_size,
        data->tokens_data,
        data->tokens_size,
        opt->iter,
        data->tokens_input,
        data->target_logits,
        data->target_probs);

    data->shuffle_countdown -= n_batch;
}

int main(int argc, char ** argv) {
    struct train_params params = get_default_train_params();

    if (!train_params_parse(argc, argv, &params)) {
        return 1;
    }

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }
    printf("%s: seed: %u\n", __func__, params.seed);
    srand(params.seed);

    struct llama_context_params llama_params = llama_context_default_params();
    llama_params.vocab_only = true;

    struct llama_model * lmodel = llama_load_model_from_file(params.fn_vocab_model, llama_params);
    struct llama_context * lctx = llama_new_context_with_model(lmodel, llama_params);

    struct llama_vocab vocab;
    {
        std::vector<const char *> strings;
        std::vector<float> scores;
        int n_vocab = llama_n_vocab(lctx);
        strings.resize(n_vocab, NULL);
        scores.resize(n_vocab, 0);
        n_vocab = llama_get_vocab(lctx, strings.data(), scores.data(), n_vocab);
        GGML_ASSERT(n_vocab == llama_n_vocab(lctx));
        vocab.id_to_token.resize(n_vocab);
        for (int i=0; i<n_vocab; ++i) {
            std::string tok   = std::string(strings[i]);
            float       score = scores[i];
            vocab.id_to_token[i].tok   = tok;
            vocab.id_to_token[i].score = score;
            vocab.token_to_id.emplace(tok, i);
        }
    }

    printf("%s: tokenize training data\n", __func__);
    std::vector<llama_token> train_tokens;
    if (tokenize_file(lctx, params.fn_train_data, train_tokens) < 0) {
        fprintf(stderr, "%s: failed to tokenize file '%s'\n", __func__, params.fn_train_data);
    }
    printf("%s: number of training tokens: %d\n", __func__, (int) train_tokens.size());

    struct my_llama_model model;
    model.hparams.n_vocab = llama_n_vocab(lctx);
    model.hparams.n_ctx   = params.n_ctx;
    model.hparams.n_embd  = params.n_embd;
    model.hparams.n_mult  = params.n_mult;
    model.hparams.n_head  = params.n_head;
    model.hparams.n_layer = params.n_layer;
    model.hparams.n_rot   = std::min((uint32_t)params.n_rotmax, model.hparams.n_embd / model.hparams.n_head);

    print_params(&model.hparams);

    std::vector<size_t> token_noccurs;
    std::vector<bool>   token_notavail;
    token_noccurs.resize(model.hparams.n_vocab, 0);
    token_notavail.resize(model.hparams.n_vocab, true);
    for (int i = 0; i < (int) train_tokens.size(); ++i) {
        ++token_noccurs[train_tokens[i]];
        token_notavail[train_tokens[i]] = false;
    }

    std::vector<float> token_freq;
    token_freq.resize(model.hparams.n_vocab, 0);
    int n_unique_tokens = 0;
    for (int i = 0; i < (int) token_noccurs.size(); ++i) {
        token_freq[i] = (float) token_noccurs[i] / (float) train_tokens.size();
        n_unique_tokens += (token_noccurs[i] > 0) ? 1 : 0;
    }
    printf("%s: number of unique tokens: %d\n", __func__, n_unique_tokens);

    struct my_llama_kv_cache kv_self;


    struct ggml_init_params lcparams;
    lcparams.mem_size   = 1024ll*1024ll*1024ll*((size_t) params.mem_model_gb);
    lcparams.mem_buffer = NULL;
    lcparams.no_alloc   = false;

    model.ctx = ggml_init(lcparams);
    kv_self.ctx = model.ctx;

    my_llama_sampler sampler;


    int n_tokens = model.hparams.n_ctx;
    int n_vocab  = model.hparams.n_vocab;
    int n_batch  = params.n_batch;

    struct ggml_opt_context * opt = (struct ggml_opt_context *) alloca(sizeof(struct ggml_opt_context));
    memset(opt, 0, sizeof(struct ggml_opt_context));

    struct ggml_opt_params opt_params_adam = ggml_opt_default_params(GGML_OPT_ADAM);
    struct ggml_opt_params opt_params_lbfgs = ggml_opt_default_params(GGML_OPT_LBFGS);
    opt_params_adam.print_forward_graph  = false;
    opt_params_adam.print_backward_graph = false;
    opt_params_adam.n_threads            = params.n_threads;
    opt_params_adam.past                 = params.opt_past;
    opt_params_adam.delta                = params.opt_delta;
    opt_params_adam.max_no_improvement   = params.opt_max_no_improvement;
    opt_params_adam.adam.n_iter          = params.adam_n_iter;
    opt_params_adam.adam.sched           = 1.0f;
    opt_params_adam.adam.alpha           = params.adam_alpha;
    opt_params_adam.adam.decay           = params.adam_decay;
    opt_params_adam.adam.decay_min_ndim  = params.adam_decay_min_ndim;
    opt_params_adam.adam.beta1           = params.adam_beta1;
    opt_params_adam.adam.beta2           = params.adam_beta2;
    opt_params_adam.adam.gclip           = params.adam_gclip;
    opt_params_adam.adam.eps_f           = params.adam_eps_f;

    opt_params_lbfgs.print_forward_graph  = false;
    opt_params_lbfgs.print_backward_graph = false;
    opt_params_lbfgs.n_threads            = params.n_threads;
    opt_params_adam.past                  = params.opt_past;
    opt_params_adam.delta                 = params.opt_delta;
    opt_params_adam.max_no_improvement    = params.opt_max_no_improvement;
    opt_params_lbfgs.lbfgs.n_iter         = params.lbfgs_n_iter;

    opt->ctx = model.ctx;
    opt->params = params.use_adam ? opt_params_adam : opt_params_lbfgs;

    printf("%s: init model\n", __func__);
    bool existed = load_checkpoint(&model, opt, params.fn_checkpoint_in, true);
    set_param_model(&model);

    opt->params = params.use_adam ? opt_params_adam : opt_params_lbfgs;

    opt->iter = model.train_its;
    printf("%s: opt iter %d\n", __func__, opt->iter);

    bool from_scratch = !existed;
    if (from_scratch) {
        randomize_model(&model, params.seed, 0.0f, 1.0f, -1.0f, +1.0f);
    }

    init_kv_cache(&kv_self, &model, 1);
    // init_kv_cache(&kv_self, &model, n_batch);
    init_sampler(&sampler, lctx);

    printf("used_mem model+cache: %zu bytes\n", ggml_used_mem(model.ctx));
    // ggml_print_tensor_objects(model.ctx);

    // TODO: use std::vector<uint8_t> intead of "new"
    size_t    compute_size = 1024ll*1024ll*1024ll*((size_t) params.mem_compute_gb);
    uint8_t * compute_addr = new uint8_t[compute_size];

    size_t size_buf_0 = 1024ll*1024ll*1024ll*((size_t) params.mem_compute0_gb);
    uint8_t * compute_buf_0 = new uint8_t[size_buf_0];

    ggml_allocr * alloc = NULL;
    if (params.use_alloc) {
        static const size_t tensor_alignment = 32;
        alloc = ggml_allocr_new(compute_buf_0, size_buf_0, tensor_alignment);
    }

    GGML_ASSERT(n_tokens < (int) train_tokens.size());
    std::vector<int> train_samples;
    train_samples.push_back(0);
    for (int i = 1; i < (int) train_tokens.size() - n_tokens; ++i) {
        if (!params.samples_start_after_nl || (train_tokens[i-1] == llama_token_nl())) {
            train_samples.push_back(i);
        }
    }
    shuffle_ints(train_samples.data(), train_samples.data() + train_samples.size());
    for (int i = 0; i < (int) train_samples.size(); ++i) {
        GGML_ASSERT(train_samples[i]+n_tokens-1 < (int) train_tokens.size());
    }

    std::vector<uint8_t> work_buffer;

    printf("%s: begin training\n", __func__);

    struct opt_callback_data opt_cb_data;
    opt_cb_data.params = &params;
    opt_cb_data.opt = opt;
    opt_cb_data.tokens_data = train_tokens.data();
    opt_cb_data.tokens_size = train_tokens.size();
    opt_cb_data.samples_data = train_samples.data();
    opt_cb_data.samples_size = train_samples.size();
    opt_cb_data.shuffle_countdown = train_samples.size();
    opt_cb_data.tokens_input  = NULL;
    opt_cb_data.target_logits = NULL;
    opt_cb_data.target_probs  = NULL;

    int64_t t0 = ggml_time_ms();

    for (int ex = 0; ex < params.n_examples; ++ex) {
        if (ex*n_batch >= (int) train_samples.size()) {
            shuffle_ints(train_samples.data(), train_samples.data() + train_samples.size());
            for (int i = 0; i < (int) train_samples.size(); ++i) {
                GGML_ASSERT(train_samples[i]+n_tokens-1 < (int) train_tokens.size());
            }
        }

        struct ggml_init_params cparams = {
            compute_size, // mem_size
            compute_addr, // mem_buffer
            false,        // no_alloc
        };
        struct ggml_context * ctx0 = ggml_init(cparams);

        ggml_set_no_alloc(ctx0, false);

        // don't use alloc for input tensors, so we can safely fill them with data
        struct ggml_tensor * after_opt_best_samples = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_batch);
        //struct ggml_tensor * after_opt_probs        = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);
        struct ggml_tensor * tokens_input           = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_batch);
        struct ggml_tensor * target_logits          = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);
        struct ggml_tensor * target_probs           = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);

        ggml_set_no_alloc(ctx0, (alloc != NULL));

        if (alloc) {
            ggml_allocr_reset(alloc);
        }

        opt_cb_data.tokens_input  = tokens_input;
        opt_cb_data.target_logits = target_logits;
        opt_cb_data.target_probs  = target_probs;

        int n_past = 0;

        struct ggml_cgraph * gf = ggml_new_graph(ctx0);
        struct ggml_cgraph * gb = ggml_new_graph(ctx0);
        struct ggml_cgraph * gb_tmp = params.use_checkpointing
            ? ggml_new_graph(ctx0)
            : NULL;

        GGML_ASSERT(n_past == 0);

        struct ggml_tensor * loss   = NULL;
        struct ggml_tensor * logits = NULL;

        loss = llama_build_train_graphs(
            &model, alloc, ctx0,
            gf, gb, gb_tmp,
            &logits, tokens_input, target_probs,
            n_tokens, n_batch,
            params.use_flash,
            params.use_checkpointing
        );

        size_t used_mem_before_opt = ggml_used_mem(ctx0);

        opt->params.adam.sched = (opt->iter < params.warmup)
            ? (float) opt->iter / (float) params.warmup
            : cosine_decay_restart(
                params.cos_decay_steps,
                params.cos_decay_min,
                opt->iter - params.warmup,
                params.cos_decay_restart,
                params.enable_restart);

        float min_sched = params.adam_min_alpha / params.adam_alpha;
        opt->params.adam.sched = min_sched + opt->params.adam.sched * (1.0f - min_sched);

        printf("%s: opt->params.adam.sched %.5f\n", __func__, opt->params.adam.sched);

        ggml_opt_resume_g(ctx0, opt, loss, gf, gb, &opt_callback, (void *) &opt_cb_data);

        size_t used_mem_after_opt = ggml_used_mem(ctx0);

        int n_iter = params.use_adam ? params.adam_n_iter : params.lbfgs_n_iter;
        model.train_its = opt->iter;
        model.train_samples += n_batch * n_iter;
        model.train_tokens  += n_batch * n_tokens * n_iter;

        if (params.print_info_interval > 0 && ex % params.print_info_interval == 0) {
            printf("Example %d, opt iter %d\n", ex, opt->iter);
            printf("error_before_opt: %.6f\n", opt->loss_before);
            printf("error_after_opt:  %.6f\n", opt->loss_after);
            printf("used_mem_before_opt: %zu bytes\n", used_mem_before_opt);
            printf("used_mem_after_opt:  %zu bytes\n", used_mem_after_opt);
        }

        if (params.print_details_interval > 0 && ex % params.print_details_interval == 0) {
            // set_logits_masked(logits, token_notavail, -1e9);
            for (int i=0; i<n_batch; ++i) {
                init_sampler(&sampler, lctx);
                for (int k=0; k<n_tokens; ++k) {
                    int32_t token = sample(&sampler,
                        (float *)       ((char *) logits->data + i*logits->nb[2] + k*logits->nb[1]),
                        (llama_token *) ((char *) tokens_input->data + i*tokens_input->nb[1]),
                        k);
                    * ((int32_t *) ((char *) after_opt_best_samples->data + i*after_opt_best_samples->nb[1] + k*after_opt_best_samples->nb[0])) = token;
                }
            }

            // printf("probabilities after optimization:\n");
            // print_matrix(after_opt_probs);
            printf("Example:\n---\n");
            print_tokens_batch(lctx, tokens_input);
            printf("\n---\n");

            // printf("best samples after optimization:\n---\n");
            printf("samples after optimization:\n---\n");
            print_tokens_batch(lctx, after_opt_best_samples);
            printf("\n---\n");
        }

        ggml_free(ctx0);
    }

    int64_t t1 = ggml_time_ms();
    int64_t d  = t1-t0;
    double  dd = (double) d * 1e-3;
    printf("%s: total training time=%f seconds\n", __func__, dd);

    if (params.n_examples > 0) {
        save_checkpoint(&model, opt, params.fn_checkpoint_out);
    }

    if (strlen(params.fn_model_out) > 0) {
        save_as_llama_model(&vocab, &model, params.fn_model_out);
    }

    {
        int n_gen = params.n_predict;
        int sample_ctx = n_tokens - n_tokens/8;

        // use defaults from common.h
        sampler.params.top_k             = 40;
        sampler.params.top_p             = 0.95f;
        sampler.params.tfs_z             = 1.00f;
        sampler.params.typical_p         = 1.00f;
        sampler.params.temp              = 0.8f;
        sampler.params.repeat_penalty    = 1.1f;
        sampler.params.repeat_last_n     = 64;
        sampler.params.frequency_penalty = 0.0f;
        sampler.params.presence_penalty  = 0.0f;
        sampler.params.mirostat          = 0;
        sampler.params.mirostat_tau      = 5.00f;
        sampler.params.mirostat_eta      = 0.10f;
        init_sampler(&sampler, lctx);

        printf("[Prediction context]\n");

        struct ggml_tensor * tokens_input  = ggml_new_tensor_1d(model.ctx, GGML_TYPE_I32, n_tokens);
        struct ggml_tensor * target_logits = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, n_vocab,  n_tokens);
        struct ggml_tensor * target_probs  = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, n_vocab,  n_tokens);

        get_example_targets(train_samples.data(), train_samples.size(), train_tokens.data(), train_tokens.size(), rand()%train_samples.size(), tokens_input, target_logits, target_probs);
        for (int i=sample_ctx; i<n_tokens; ++i) {
            ggml_set_i32_1d(tokens_input, i, n_vocab/2);
        }

        for (int i=0; i<sample_ctx-1; ++i) {
            print_token(lctx, ggml_get_i32_1d(tokens_input, i));
        }

        printf("\n[Generating %d tokens]\n", n_gen);
        for (int i=0; i<n_gen; ++i) {
            struct ggml_init_params cparams = {
                compute_size, // .mem_size
                compute_addr, // .mem_buffer
                false,        // .no_alloc
            };
            struct ggml_context * ctx0 = ggml_init(cparams);

            struct ggml_cgraph * gf = ggml_new_graph(ctx0);

            int n_past = 0;
            struct ggml_tensor * logits = forward(&model, &kv_self, ctx0, gf, tokens_input, sample_ctx, n_past);

            ggml_build_forward_expand(gf, logits);
            ggml_graph_compute_helper(work_buffer, gf, params.n_threads);

            //struct ggml_tensor * best_samples = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, sample_ctx);
            //struct ggml_tensor * probs        = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_vocab, sample_ctx);

            // set_logits_masked(logits, token_notavail, -1e9);
            int token = sample(&sampler,
                (float *) ((char *) logits->data + (sample_ctx-1)*logits->nb[1]),
                (llama_token *) tokens_input->data,
                sample_ctx-1);
            //int token = ggml_get_i32_1d(best_samples, sample_ctx-1);

            // print_row(probs, sample_at);
            print_token(lctx, token);

            lshift_examples(tokens_input, target_logits, target_probs, 1);
            ggml_set_i32_1d(tokens_input, 0, 0);
            ggml_set_i32_1d(tokens_input, sample_ctx-1, token);

            ggml_free(ctx0);
        }
    }

    if (alloc) {
        ggml_allocr_free(alloc);
    }

    delete[] compute_addr;
    delete[] compute_buf_0;
    ggml_free(model.ctx);
    llama_free(lctx);
    llama_free_model(lmodel);
    return 0;
}

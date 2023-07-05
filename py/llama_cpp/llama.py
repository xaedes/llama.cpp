import os
import sys
import glob
import ctypes

from ctypes import c_int, c_float, c_double, c_char_p, c_void_p, c_bool, c_size_t, c_ubyte, POINTER, Structure, c_int64


# Load the library
if sys.platform == 'win32':
    lib = ctypes.cdll.LoadLibrary(next(iter(glob.glob(os.path.join(os.path.dirname(__file__), '..', '..', '**', 'llama.dll'), recursive=True))))
else:
    lib = ctypes.cdll.LoadLibrary(next(iter(glob.glob(os.path.join(os.path.dirname(__file__), '..', '..', '**', 'libllama.so'), recursive=True))))


# C types
llama_token = c_int
llama_token_p = POINTER(llama_token)

class llama_token_data(Structure):
    _fields_ = [
        ('id',   llama_token), # token id
        ('p',    c_float), # probability of the token
        ('plog', c_float), # log probability of the token
    ]

llama_token_data_p = POINTER(llama_token_data)

class llama_token_data_array(Structure):
    _fields_ = [
        ('data',   llama_token_data_p),
        ('size',   c_size_t),
        ('sorted', c_bool),
    ]

llama_token_data_array_p = POINTER(llama_token_data_array)

GGML_CUDA_MAX_DEVICES = 16
LLAMA_MAX_DEVICES = GGML_CUDA_MAX_DEVICES

llama_progress_callback = ctypes.CFUNCTYPE(None, c_float, c_void_p)
class llama_context_params(Structure):
    _fields_ = [
        ('seed',         c_int),  # RNG seed, 0 for random
        ('n_ctx',        c_int),  # text context
        ('n_batch',      c_int),  # prompt processing batch size
        ('n_gpu_layers', c_int),  # number of layers to store in VRAM
        ('main_gpu',     c_int),  # the GPU that is used for scratch and small tensors
        ('tensor_split', c_float * LLAMA_MAX_DEVICES),  # how to split layers across multiple GPUs
        ('progress_callback',           llama_progress_callback), # called with a progress value between 0 and 1, pass NULL to disable
        ('progress_callback_user_data', c_void_p),                # context pointer passed to the progress callback
        ('low_vram',     c_bool), # if true, reduce VRAM usage at the cost of performance
        ('f16_kv',       c_bool), # use fp16 for KV cache
        ('logits_all',   c_bool), # the llama_eval() call computes all logits, not just the last one
        ('vocab_only',   c_bool), # only load the vocabulary, no weights
        ('use_mmap',     c_bool), # use mmap if possible
        ('use_mlock',    c_bool), # force system to keep model in RAM
        ('embedding',    c_bool), # embedding mode only
    ]
llama_context_params_p = POINTER(llama_context_params)

LLAMA_FTYPE_ALL_F32              = 0
LLAMA_FTYPE_MOSTLY_F16           = 1
LLAMA_FTYPE_MOSTLY_Q4_0          = 2
LLAMA_FTYPE_MOSTLY_Q4_1          = 3
LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4
# LLAMA_FTYPE_MOSTLY_Q4_2       = 5   # support has been removed
# LLAMA_FTYPE_MOSTLY_Q4_3       = 6   # support has been removed
LLAMA_FTYPE_MOSTLY_Q8_0          = 7
LLAMA_FTYPE_MOSTLY_Q5_0          = 8
LLAMA_FTYPE_MOSTLY_Q5_1          = 9
LLAMA_FTYPE_MOSTLY_Q2_K          = 10
LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11
LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12
LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13
LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14
LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15
LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16
LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17
LLAMA_FTYPE_MOSTLY_Q6_K          = 18

class llama_model_quantize_params(Structure):
    _fields_ = [
        ('nthread',      c_int),  # number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        ('ftype',        c_int),  # quantize to this llama_ftype LLAMA_FTYPE_*
        ('allow_requantize',       c_bool), # allow quantizing non-f32/f16 tensors
        ('quantize_output_tensor', c_bool), # quantize output.weight
    ]

llama_model_quantize_params_p = POINTER(llama_model_quantize_params)

llama_context_p = c_void_p
llama_model_p = c_void_p

c_size_p = POINTER(c_size_t)
c_ubyte_p = POINTER(c_ubyte)
c_float_p = POINTER(c_float)

# C functions
lib.llama_context_default_params.argtypes = []
lib.llama_context_default_params.restype = llama_context_params

lib.llama_model_quantize_default_params.argtypes = []
lib.llama_model_quantize_default_params.restype = llama_model_quantize_params

lib.llama_mmap_supported.argtypes = []
lib.llama_mmap_supported.restype = c_bool

lib.llama_mlock_supported.argtypes = []
lib.llama_mlock_supported.restype = c_bool

lib.llama_init_backend.argtypes = [c_bool]
lib.llama_init_backend.restype = None

lib.llama_time_us.argtypes = [c_bool]
lib.llama_time_us.restype = c_int64

lib.llama_load_model_from_file.argtypes = [c_char_p, llama_context_params]
lib.llama_load_model_from_file.restype = llama_model_p

lib.llama_free_model.argtypes = [llama_model_p]
lib.llama_free_model.restype = None

lib.llama_new_context_with_model.argtypes = [llama_model_p, llama_context_params]
lib.llama_new_context_with_model.restype = llama_context_p

# DEPRECATED - please use llama_load_model_from_file combined with llama_new_context_with_model
lib.llama_init_from_file.argtypes = [c_char_p, llama_context_params]
lib.llama_init_from_file.restype = llama_context_p

lib.llama_free.argtypes = [llama_context_p]
lib.llama_free.restype = None

lib.llama_model_quantize.argtypes = [c_char_p, c_char_p, llama_model_quantize_params_p]
lib.llama_model_quantize.restype = c_int

# DEPRECATED - please use llama_model_apply_lora_from_file instead
lib.llama_apply_lora_from_file.argtypes = [llama_context_p, c_char_p, c_char_p, c_int]
lib.llama_apply_lora_from_file.restype = c_int

lib.llama_model_apply_lora_from_file.argtypes = [llama_model_p, c_char_p, c_char_p, c_int]
lib.llama_model_apply_lora_from_file.restype = c_int

lib.llama_get_kv_cache_token_count.argtypes = [llama_context_p]
lib.llama_get_kv_cache_token_count.restype = c_int

lib.llama_set_rng_seed.argtypes = [llama_context_p, c_int]
lib.llama_set_rng_seed.restype = None

lib.llama_get_state_size.argtypes = [llama_context_p]
lib.llama_get_state_size.restype = c_size_t

lib.llama_copy_state_data.argtypes = [llama_context_p, c_ubyte_p]
lib.llama_copy_state_data.restype = c_size_t

lib.llama_set_state_data.argtypes = [llama_context_p, c_ubyte_p]
lib.llama_set_state_data.restype = c_size_t

lib.llama_load_session_file.argtypes = [llama_context_p, c_char_p, llama_token_p, c_size_t, c_size_p]
lib.llama_load_session_file.restype = c_bool

lib.llama_save_session_file.argtypes = [llama_context_p, c_char_p, llama_token_p, c_size_t]
lib.llama_save_session_file.restype = c_bool

lib.llama_eval.argtypes = [llama_context_p, llama_token_p, c_int, c_int, c_int]
lib.llama_eval.restype = c_int

lib.llama_eval_embd.argtypes = [llama_context_p, c_float_p, c_int, c_int, c_int]
lib.llama_eval_embd.restype = c_int

lib.llama_eval_export.argtypes = [llama_context_p, c_char_p]
lib.llama_eval_export.restype = c_int

lib.llama_tokenize.argtypes = [llama_context_p, c_char_p, llama_token_p, c_int, c_bool]
lib.llama_tokenize.restype = c_int

lib.llama_n_vocab.argtypes = [llama_context_p]
lib.llama_n_vocab.restype = c_int

lib.llama_n_ctx.argtypes = [llama_context_p]
lib.llama_n_ctx.restype = c_int

lib.llama_n_embd.argtypes = [llama_context_p]
lib.llama_n_embd.restype = c_int

lib.llama_get_vocab.argtypes = [llama_context_p, POINTER(c_char_p), c_float_p, c_int]
lib.llama_get_vocab.restype = c_int

lib.llama_get_logits.argtypes = [llama_context_p]
lib.llama_get_logits.restype = c_float_p

lib.llama_get_embeddings.argtypes = [llama_context_p]
lib.llama_get_embeddings.restype = c_float_p

lib.llama_token_to_str.argtypes = [llama_context_p, llama_token]
lib.llama_token_to_str.restype = c_char_p

lib.llama_token_bos.argtypes = []
lib.llama_token_bos.restype = llama_token

lib.llama_token_eos.argtypes = []
lib.llama_token_eos.restype = llama_token

lib.llama_token_nl.argtypes = []
lib.llama_token_nl.restype = llama_token

lib.llama_sample_repetition_penalty.argtypes = [llama_context_p, llama_token_data_array_p, llama_token_p, c_size_t, c_float]
lib.llama_sample_repetition_penalty.restype = None

lib.llama_sample_frequency_and_presence_penalties.argtypes = [llama_context_p, llama_token_data_array_p, llama_token_p, c_size_t, c_float, c_float]
lib.llama_sample_frequency_and_presence_penalties.restype = None

lib.llama_sample_softmax.argtypes = [llama_context_p, llama_token_data_array_p]
lib.llama_sample_softmax.restype = None

lib.llama_sample_top_k.argtypes = [llama_context_p, llama_token_data_array_p, c_int, c_size_t]
lib.llama_sample_top_k.restype = None

lib.llama_sample_top_p.argtypes = [llama_context_p, llama_token_data_array_p, c_float, c_size_t]
lib.llama_sample_top_p.restype = None

lib.llama_sample_tail_free.argtypes = [llama_context_p, llama_token_data_array_p, c_float, c_size_t]
lib.llama_sample_tail_free.restype = None

lib.llama_sample_typical.argtypes = [llama_context_p, llama_token_data_array_p, c_float, c_size_t]
lib.llama_sample_typical.restype = None

lib.llama_sample_temperature.argtypes = [llama_context_p, llama_token_data_array_p, c_float]
lib.llama_sample_temperature.restype = None

lib.llama_sample_token_mirostat.argtypes = [llama_context_p, llama_token_data_array_p, c_float, c_float, c_int, c_float_p]
lib.llama_sample_token_mirostat.restype = llama_token

lib.llama_sample_token_mirostat_v2.argtypes = [llama_context_p, llama_token_data_array_p, c_float, c_float, c_float_p]
lib.llama_sample_token_mirostat_v2.restype = llama_token

lib.llama_sample_token_greedy.argtypes = [llama_context_p, llama_token_data_array_p]
lib.llama_sample_token_greedy.restype = llama_token

lib.llama_sample_token.argtypes = [llama_context_p, llama_token_data_array_p]
lib.llama_sample_token.restype = llama_token

lib.llama_print_timings.argtypes = [llama_context_p]
lib.llama_print_timings.restype = None

lib.llama_reset_timings.argtypes = [llama_context_p]
lib.llama_reset_timings.restype = None

lib.llama_print_system_info.argtypes = []
lib.llama_print_system_info.restype = c_char_p


# Python functions
def llama_context_default_params() -> llama_context_params:
    params = lib.llama_context_default_params()
    return params

def llama_model_quantize_default_params() -> llama_model_quantize_params:
    params = lib.llama_model_quantize_default_params()
    return params

def llama_mmap_supported() -> bool:
    return lib.llama_mmap_supported()

def llama_mlock_supported() -> bool:
    return lib.llama_mlock_supported()

def llama_init_backend(numa:bool):
    """TODO: not great API - very likely to change
    Initialize the llama + ggml backend
    If numa is true, use NUMA optimizations
    Call once at the start of the program"""
    lib.llama_init_backend(bool(numa))

def llama_time_us() -> int:
    return lib.llama_time_us()

def llama_load_model_from_file(path_model: str, params: llama_context_params) -> llama_model_p:
    return lib.llama_load_model_from_file(path_model.encode('utf-8'), params)

def llama_free_model(model: llama_model_p):
    lib.llama_free_model(ctx)

def llama_new_context_with_model(model: llama_model_p, params: llama_context_params) -> llama_context_p:
    return lib.llama_new_context_with_model(model, params)

def llama_init_from_file(path_model: str, params: llama_context_params) -> llama_context_p:
    """DEPRECATED: please use llama_load_model_from_file combined with llama_new_context_with_model instead"""
    return lib.llama_init_from_file(path_model.encode('utf-8'), params)

def llama_free(ctx: llama_context_p):
    """Free all allocated memory"""
    lib.llama_free(ctx)

def llama_model_quantize(fname_inp: str, fname_out: str, params: llama_model_quantize_params_p) -> c_int:
    """Returns 0 on success"""
    return lib.llama_model_quantize(fname_inp.encode('utf-8'), fname_out.encode('utf-8'), params)

def llama_apply_lora_from_file(ctx: llama_context_p, path_lora: str, path_base_model: str, n_threads: c_int) -> c_int:
    """DEPRECATED: please use llama_model_apply_lora_from_file instead"""
    return lib.llama_apply_lora_from_file(ctx, path_lora.encode('utf-8'), path_base_model.encode('utf-8'), n_threads)

def llama_model_apply_lora_from_file(model: llama_model_p, path_lora: str, path_base_model: str, n_threads: c_int) -> c_int:
    return lib.llama_model_apply_lora_from_file(model, path_lora.encode('utf-8'), path_base_model.encode('utf-8'), n_threads)

def llama_get_kv_cache_token_count(ctx: llama_context_p) -> c_int:
    return lib.llama_get_kv_cache_token_count(ctx)

def llama_set_rng_seed(ctx: llama_context_p, seed: c_int):
    return lib.llama_set_rng_seed(ctx, seed)

def llama_get_state_size(ctx: llama_context_p) -> c_size_t:
    return lib.llama_get_state_size(ctx)

def llama_copy_state_data(ctx: llama_context_p, dst: c_ubyte_p) -> c_size_t:
    return lib.llama_copy_state_data(ctx, dst)

def llama_set_state_data(ctx: llama_context_p, src: c_ubyte_p) -> c_size_t:
    return lib.llama_set_state_data(ctx, src)

def llama_load_session_file(ctx: llama_context_p, path_session: str, tokens_out: llama_token_p, n_token_capacity: c_size_t, n_token_count_out: c_size_p) -> c_bool:
    return lib.llama_load_session_file(ctx, path_session.encode('utf-8'), tokens_out, n_token_capacity, n_token_count_out)

def llama_save_session_file(ctx: llama_context_p, path_session: str, tokens: llama_token_p, n_token_count: c_size_t) -> c_bool:
    return lib.llama_save_session_file(ctx, path_session.encode('utf-8'), tokens, n_token_count)

def llama_eval(ctx: llama_context_p, tokens: llama_token_p, n_tokens: c_int, n_past: c_int, n_threads: c_int) -> c_int:
    """Run the llama inference to obtain the logits and probabilities for the next token.
    tokens + n_tokens is the provided batch of new tokens to process
    n_past is the number of tokens to use from previous eval calls
    Returns 0 on success"""
    return lib.llama_eval(ctx, tokens, n_tokens, n_past, n_threads)

def llama_eval_embd(ctx: llama_context_p, embd: c_float_p, n_tokens: c_int, n_past: c_int, n_threads: c_int) -> c_int:
    """Same as llama_eval, but use float matrix input directly."""
    return lib.llama_eval_embd(ctx, embd, n_tokens, n_past, n_threads)

def llama_eval_export(ctx: llama_context_p, fname: str) -> c_int:
    """Export a static computation graph for context of 511 and batch size of 1
    NOTE: since this functionality is mostly for debugging and demonstration purposes, we hardcode these
          parameters here to keep things simple
    IMPORTANT: do not use for anything else other than debugging and testing!"""
    return lib.llama_eval_export(ctx, fname.encode('utf-8'))

def llama_tokenize(ctx: llama_context_p, text: str, tokens: llama_token_p, n_max_tokens: c_int, add_bos: c_bool) -> c_int:
    """Convert the provided text into tokens.
    The tokens pointer must be large enough to hold the resulting tokens.
    Returns the number of tokens on success, no more than n_max_tokens
    Returns a negative number on failure - the number of tokens that would have been returned"""
    return lib.llama_tokenize(ctx, text.encode('utf-8'), tokens, n_max_tokens, add_bos)

def llama_get_vocab(ctx: llama_context_p, strings: POINTER(c_char_p), scores: c_float_p, capacity: int) -> c_int:
    """Get the vocabulary as output parameters.
    Returns number of results."""
    return lib.llama_get_vocab(ctx, strings, scores, capacity)

def llama_n_vocab(ctx: llama_context_p) -> c_int:
    return lib.llama_n_vocab(ctx)

def llama_n_ctx(ctx: llama_context_p) -> c_int:
    return lib.llama_n_ctx(ctx)

def llama_n_embd(ctx: llama_context_p) -> c_int:
    return lib.llama_n_embd(ctx)

def llama_get_logits(ctx: llama_context_p) -> c_float_p:
    """Token logits obtained from the last call to llama_eval()
    The logits for the last token are stored in the last row
    Can be mutated in order to change the probabilities of the next token
    Rows: n_tokens
    Cols: n_vocab"""
    return lib.llama_get_logits(ctx)

def llama_get_embeddings(ctx: llama_context_p) -> c_float_p:
    """Get the embeddings for the input
    shape: [n_embd] (1-dimensional)"""
    return lib.llama_get_embeddings(ctx)

def llama_token_to_str(ctx: llama_context_p, token: int) -> str:
    """Token Id -> String. Uses the vocabulary in the provided context"""
    return lib.llama_token_to_str(ctx, token).decode('utf-8', errors='ignore')

def llama_token_bos() -> llama_token:
    return lib.llama_token_bos()

def llama_token_eos() -> llama_token:
    return lib.llama_token_eos()

def llama_token_nl() -> llama_token:
    return lib.llama_token_nl()

def llama_sample_repetition_penalty(ctx: llama_context_p, candidates: llama_token_data_array_p, last_tokens: llama_token_p, last_tokens_size: c_size_t, penalty: float):
    lib.llama_sample_repetition_penalty(ctx, candidates, last_tokens, last_tokens_size, penalty)

def llama_sample_frequency_and_presence_penalties(ctx: llama_context_p, candidates: llama_token_data_array_p, last_tokens: llama_token_p, last_tokens_size: c_size_t, alpha_frequency: float, alpha_presence: float):
    lib.llama_sample_frequency_and_presence_penalties(ctx, candidates, last_tokens, last_tokens_size, alpha_frequency, alpha_presence)

def llama_sample_softmax(ctx: llama_context_p, candidates: llama_token_data_array_p):
    lib.llama_sample_softmax(ctx, candidates)

def llama_sample_top_k(ctx: llama_context_p, candidates: llama_token_data_array_p, k: c_int, min_keep: c_size_t):
    lib.llama_sample_top_k(ctx, candidates, k, min_keep)

def llama_sample_top_p(ctx: llama_context_p, candidates: llama_token_data_array_p, p: float, min_keep: c_size_t):
    lib.llama_sample_top_p(ctx, candidates, c_float(p), c_size_t(min_keep))

def llama_sample_tail_free(ctx: llama_context_p, candidates: llama_token_data_array_p, z: float, min_keep: c_size_t):
    lib.llama_sample_tail_free(ctx, candidates, z, min_keep)

def llama_sample_typical(ctx: llama_context_p, candidates: llama_token_data_array_p, p: float, min_keep: c_size_t):
    lib.llama_sample_typical(ctx, candidates, p, min_keep)

def llama_sample_temperature(ctx: llama_context_p, candidates: llama_token_data_array_p, temp: float):
    lib.llama_sample_temperature(ctx, candidates, temp)

def llama_sample_token_mirostat(ctx: llama_context_p, candidates: llama_token_data_array_p, tau: float, eta: float, m: c_int, mu: c_float_p) -> llama_token:
    return lib.llama_sample_token_mirostat(ctx, candidates, tau, eta, m, mu)

def llama_sample_token_mirostat_v2(ctx: llama_context_p, candidates: llama_token_data_array_p, tau: float, eta: float, mu: c_float_p) -> llama_token:
    return lib.llama_sample_token_mirostat_v2(ctx, candidates, tau, eta, mu)

def llama_sample_token_greedy(ctx: llama_context_p, candidates: llama_token_data_array_p) -> llama_token:
    return lib.llama_sample_token_greedy(ctx, candidates)

def llama_sample_token(ctx: llama_context_p, candidates: llama_token_data_array_p) -> llama_token:
    return lib.llama_sample_token(ctx, candidates)

def llama_print_timings(ctx: llama_context_p):
    lib.llama_print_timings(ctx)

def llama_reset_timings(ctx: llama_context_p):
    lib.llama_reset_timings(ctx)

def llama_print_system_info() -> c_char_p:
    return lib.llama_print_system_info()

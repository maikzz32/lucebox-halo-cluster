#include "common/gguf_shards.h"

#include "common/gguf_bounds.h"

#include <cinttypes>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <set>

namespace dflash::common {

namespace {

// Match the tail "-NNNNN-of-MMMMM.gguf" that gguf-split writes -- fifteen
// characters before the extension. Returns false when the name is not a split
// part, which is the common case and not an error.
bool parse_split_name(const std::string & path, std::string & prefix,
                      int & part, int & count) {
    static const char kExt[] = ".gguf";
    const size_t ext_len  = sizeof(kExt) - 1;          // 5
    const size_t tail_len = 15;                        // -00001-of-00003
    if (path.size() < ext_len + tail_len) return false;
    if (path.compare(path.size() - ext_len, ext_len, kExt) != 0) return false;

    const size_t tail_begin = path.size() - ext_len - tail_len;
    const std::string tail = path.substr(tail_begin, tail_len);
    if (tail[0] != '-') return false;
    if (tail.compare(6, 4, "-of-") != 0) return false;
    for (size_t i = 1; i <= 5; ++i)  if (tail[i] < '0' || tail[i] > '9') return false;
    for (size_t i = 10; i <= 14; ++i) if (tail[i] < '0' || tail[i] > '9') return false;

    part   = std::atoi(tail.substr(1, 5).c_str());
    count  = std::atoi(tail.substr(10, 5).c_str());
    prefix = path.substr(0, tail_begin);
    return part >= 1 && count >= 1 && part <= count;
}

std::string split_part_name(const std::string & prefix, int part, int count) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "-%05d-of-%05d.gguf", part, count);
    return prefix + buf;
}

const char * kv_str(gguf_context * gctx, const char * key, const char * fallback) {
    const int64_t id = gguf_find_key(gctx, key);
    if (id < 0) return fallback;
    if (gguf_get_kv_type(gctx, id) != GGUF_TYPE_STRING) return fallback;
    return gguf_get_val_str(gctx, id);
}

// Split bookkeeping keys, absent in an unsplit file.
bool kv_u16(gguf_context * gctx, const char * key, uint32_t & out) {
    const int64_t id = gguf_find_key(gctx, key);
    if (id < 0) return false;
    switch (gguf_get_kv_type(gctx, id)) {
        case GGUF_TYPE_UINT16: out = gguf_get_val_u16(gctx, id); return true;
        case GGUF_TYPE_UINT32: out = gguf_get_val_u32(gctx, id); return true;
        case GGUF_TYPE_INT16:  out = (uint32_t) gguf_get_val_i16(gctx, id); return true;
        case GGUF_TYPE_INT32:  out = (uint32_t) gguf_get_val_i32(gctx, id); return true;
        default: return false;
    }
}

}  // namespace

GgufShardSet::~GgufShardSet() {
    for (Shard & s : shards_) {
        if (s.gctx) gguf_free(s.gctx);
        if (s.meta_ctx) ggml_free(s.meta_ctx);
    }
    shards_.clear();
    index_.clear();
}

std::vector<std::string> GgufShardSet::sibling_paths(const std::string & path) {
    std::string prefix;
    int part = 0, count = 0;
    if (!parse_split_name(path, prefix, part, count)) return {path};
    std::vector<std::string> out;
    out.reserve((size_t) count);
    for (int i = 1; i <= count; ++i) out.push_back(split_part_name(prefix, i, count));
    return out;
}

bool GgufShardSet::open(const std::string & path, std::string & err) {
    this->~GgufShardSet();
    new (this) GgufShardSet();

    const std::vector<std::string> paths = sibling_paths(path);
    shards_.resize(paths.size());

    for (size_t i = 0; i < paths.size(); ++i) {
        Shard & s = shards_[i];
        s.path = paths[i];

        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = &s.meta_ctx;
        s.gctx = gguf_init_from_file(s.path.c_str(), params);
        if (!s.gctx) {
            err = "gguf shard set: cannot read part " + std::to_string(i + 1) + " of " +
                  std::to_string(paths.size()) + ": " + s.path +
                  (paths.size() > 1 ? " (a split GGUF needs every part beside the first)" : "");
            this->~GgufShardSet();
            new (this) GgufShardSet();
            return false;
        }
        if (!s.map.open(s.path, err)) {
            this->~GgufShardSet();
            new (this) GgufShardSet();
            return false;
        }
        s.data_off = gguf_get_data_offset(s.gctx);

        const int64_t n = gguf_get_n_tensors(s.gctx);
        for (int64_t t = 0; t < n; ++t) {
            const char * name = gguf_get_tensor_name(s.gctx, t);
            if (!name) continue;
            auto it = index_.find(name);
            if (it != index_.end()) {
                err = std::string("gguf shard set: tensor '") + name +
                      "' appears in part " + std::to_string(it->second.first + 1) +
                      " and again in part " + std::to_string(i + 1) +
                      "; the parts do not belong to one model";
                this->~GgufShardSet();
                new (this) GgufShardSet();
                return false;
            }
            index_.emplace(name, std::make_pair((int) i, t));
        }
    }

    if (!validate(err)) {
        this->~GgufShardSet();
        new (this) GgufShardSet();
        return false;
    }
    return true;
}

// What is checked, and why each check earns its place:
//   - every part agrees on general.architecture: parts from two different
//     models in one directory would otherwise merge silently
//   - split.no covers 0..N-1 exactly once: a missing or duplicated part
//     otherwise shows up as absent tensors much later
//   - split.count agrees with the number of files the name implied
//   - split.tensors.count, when present, equals what we actually indexed
bool GgufShardSet::validate(std::string & err) const {
    if (shards_.empty()) { err = "gguf shard set: no parts opened"; return false; }

    // Only the first part carries the metadata. gguf-split writes the full
    // key/value block once and gives the remaining parts three keys -- split.no,
    // split.count, split.tensors.count -- and nothing else. Measured on the
    // qwen4exp model: 66 keys in part 1, 3 in each of parts 2 and 3. So the
    // architecture is compared only against parts that declare one, and part 1
    // must be the one that does.
    const std::string arch0 = kv_str(shards_[0].gctx, "general.architecture", "");
    if (arch0.empty()) {
        err = "gguf shard set: part 1 carries no general.architecture; either it is "
              "not the first part of the split, or the file is not a model";
        return false;
    }
    for (size_t i = 1; i < shards_.size(); ++i) {
        const std::string arch = kv_str(shards_[i].gctx, "general.architecture", "");
        if (!arch.empty() && arch != arch0) {
            err = "gguf shard set: part 1 is architecture '" + arch0 + "' but part " +
                  std::to_string(i + 1) + " is '" + arch + "'";
            return false;
        }
    }

    if (shards_.size() == 1) return true;   // unsplit file: nothing further to check

    std::set<uint32_t> seen;
    for (size_t i = 0; i < shards_.size(); ++i) {
        uint32_t no = 0, count = 0;
        if (!kv_u16(shards_[i].gctx, "split.no", no)) {
            err = "gguf shard set: part " + std::to_string(i + 1) +
                  " has no split.no key, so it is not a part of a split model";
            return false;
        }
        if (kv_u16(shards_[i].gctx, "split.count", count) && count != (uint32_t) shards_.size()) {
            err = "gguf shard set: part " + std::to_string(i + 1) + " says the model has " +
                  std::to_string(count) + " parts, but the file name implies " +
                  std::to_string(shards_.size());
            return false;
        }
        if (!seen.insert(no).second) {
            err = "gguf shard set: split.no " + std::to_string(no) + " appears twice";
            return false;
        }
    }
    for (uint32_t i = 0; i < (uint32_t) shards_.size(); ++i) {
        if (!seen.count(i)) {
            err = "gguf shard set: no part carries split.no " + std::to_string(i);
            return false;
        }
    }

    uint32_t declared = 0;
    if (kv_u16(shards_[0].gctx, "split.tensors.count", declared) &&
        declared != (uint32_t) index_.size()) {
        err = "gguf shard set: the parts declare " + std::to_string(declared) +
              " tensors but " + std::to_string(index_.size()) + " were found";
        return false;
    }
    return true;
}

bool GgufShardSet::find(const char * name, GgufShardTensor & out, std::string & err) const {
    out = GgufShardTensor{};
    if (!name) { err = "gguf shard set: null tensor name"; return false; }
    auto it = index_.find(name);
    if (it == index_.end()) {
        err = std::string("gguf shard set: tensor '") + name + "' not found in any of the " +
              std::to_string(shards_.size()) + " part(s)";
        return false;
    }
    const int shard_idx = it->second.first;
    const int64_t tid   = it->second.second;
    const Shard & s = shards_[(size_t) shard_idx];

    const size_t rel  = gguf_get_tensor_offset(s.gctx, tid);
    const size_t size = gguf_get_tensor_size(s.gctx, tid);

    // Checked against THIS shard's file size, never against a sum: the whole
    // point of the invariant in gguf_bounds.h is that it bounds one mapping.
    if (!gguf_tensor_in_file(s.data_off, rel, size, s.map.size())) {
        err = gguf_bounds_error("gguf shard set (part " + std::to_string(shard_idx + 1) + ")",
                                name,
                                ggml_type_name(ggml_get_tensor(s.meta_ctx, name)
                                                   ? ggml_get_tensor(s.meta_ctx, name)->type
                                                   : GGML_TYPE_COUNT),
                                s.data_off, rel, size, s.map.size());
        return false;
    }

    out.shard = shard_idx;
    out.index = tid;
    out.size  = size;
    out.path  = s.path;
    out.file_offset = s.data_off + rel;
    out.data  = static_cast<const uint8_t *>(s.map.data()) + s.data_off + rel;
    out.meta  = ggml_get_tensor(s.meta_ctx, name);
    return true;
}

void GgufShardSet::advise_willneed(const GgufShardTensor & t) const {
    if (t.shard < 0 || (size_t) t.shard >= shards_.size() || !t.data) return;
    const Shard & s = shards_[(size_t) t.shard];
    const uint8_t * base = static_cast<const uint8_t *>(s.map.data());
    if (!base) return;
    s.map.advise_willneed((size_t) (static_cast<const uint8_t *>(t.data) - base), t.size);
}

std::string GgufShardSet::describe() const {
    if (shards_.empty()) return "gguf shard set: closed";
    size_t bytes = 0;
    for (const Shard & s : shards_) bytes += s.map.size();
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%zu part(s), %" PRId64 " tensors, %.1f GiB",
                  shards_.size(), (int64_t) index_.size(),
                  (double) bytes / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

}  // namespace dflash::common

// common/gguf_shards.h — read a GGUF that was written as several files.
//
// `gguf-split` writes a model as N parts named ...-00001-of-00003.gguf, each a
// complete GGUF with the full key/value metadata and its own slice of the
// tensors. The vendored gguf reader has no notion of this: gguf_init_from_file
// opens exactly one file, and gguf_tensor_in_file() (common/gguf_bounds.h)
// takes exactly one file_size. A loader handed part 1 therefore finds a third
// of the tensors and no error.
//
// GgufShardSet opens the part it is given plus every sibling, and answers
// tensor lookups across all of them. A single unsplit file is the N == 1 case,
// so a loader can use this unconditionally.
//
// The bounds invariant from gguf_bounds.h is preserved exactly, not relaxed:
// every lookup is checked against the file size of the shard that holds the
// tensor, never against a sum.
//
// Include convention: #include "common/gguf_shards.h"

#pragma once

#include "common/gguf_mmap.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace dflash::common {

// Where a tensor lives and how to read it.
struct GgufShardTensor {
    int          shard = -1;        // index into the shard list, 0-based
    int64_t      index = -1;        // tensor index inside that shard's gguf_context
    const void * data  = nullptr;   // into that shard's mapping; valid while the set is open
    size_t       size  = 0;         // bytes
    ggml_tensor * meta = nullptr;   // shape/type, owned by that shard's meta context
};

class GgufShardSet {
public:
    GgufShardSet() = default;
    ~GgufShardSet();

    GgufShardSet(const GgufShardSet &) = delete;
    GgufShardSet & operator=(const GgufShardSet &) = delete;

    // Open `path`. If its name matches the split pattern, every sibling part is
    // opened as well and the set is validated (see validate() in the .cpp for
    // what is checked). Returns false with a diagnostic in `err`.
    bool open(const std::string & path, std::string & err);

    int  shard_count() const { return (int) shards_.size(); }
    bool is_open() const { return !shards_.empty(); }

    // Metadata reader for shard 0. gguf-split duplicates the key/value block
    // into every part; open() verifies the architecture and the split keys
    // agree rather than assuming it, and callers then read hyperparameters
    // from here as they would from a single file.
    gguf_context * meta() const { return shards_.empty() ? nullptr : shards_[0].gctx; }

    // Every tensor name across every shard, in shard-then-file order.
    int64_t total_tensors() const { return (int64_t) index_.size(); }

    bool has(const char * name) const { return index_.find(name) != index_.end(); }

    // Locate a tensor, bounds-checked against its own shard's file size.
    // Returns false with a diagnostic if the name is absent or the region does
    // not lie within that shard.
    bool find(const char * name, GgufShardTensor & out, std::string & err) const;

    // Hint the kernel that a tensor's bytes are about to be read.
    void advise_willneed(const GgufShardTensor & t) const;

    // Human-readable summary for a startup banner.
    std::string describe() const;

private:
    struct Shard {
        std::string    path;
        gguf_context * gctx = nullptr;
        ggml_context * meta_ctx = nullptr;
        GgufMmap       map;
        size_t         data_off = 0;
    };

    // Expand a path into the list of parts. Returns just `path` when the name
    // does not match the split pattern.
    static std::vector<std::string> sibling_paths(const std::string & path);

    bool validate(std::string & err) const;

    std::vector<Shard> shards_;
    // name -> (shard index, tensor index within that shard)
    std::unordered_map<std::string, std::pair<int, int64_t>> index_;
};

}  // namespace dflash::common

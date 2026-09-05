#include "cluster/fast_reduce.h"

#include <hip/hip_runtime.h>
#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace dflash::cluster {

// The kernels live in fast_reduce_kernels.cu (LANGUAGE HIP).
void fast_reduce_launch_publish(const float * src, float * staging,
                                uint32_t * pub_flag, uint32_t * slot_done,
                                uint32_t seq, uint32_t slot_span, int n,
                                uint64_t spin_limit, hipStream_t stream);
void fast_reduce_launch_wait_add(float * dst, const float * const * peer_bufs,
                                 uint32_t * const * peer_flags,
                                 uint32_t * slot_done, uint32_t seq,
                                 int n_peers, int n, uint64_t spin_limit,
                                 uint32_t * timed_out, hipStream_t stream);

namespace {

// What one rank has to tell every other rank to be written into.
constexpr int kMaxRanks = 8;

struct Endpoint {
    // qpn[r] is the queue pair this rank created to talk to rank r. Every pair
    // needs its own number, so one per rank is not enough: rank 2 must connect
    // to the queue pair rank 1 made *for rank 2*, not to whichever one rank 1
    // happened to make first.
    uint32_t qpn[kMaxRanks] = {0};
    uint32_t psn = 0;
    uint32_t rkey = 0;
    uint64_t data_addr = 0;      // base of the receive region
    uint64_t flag_addr = 0;      // base of the flag region
    uint8_t  gid[16] = {0};
};

bool set_nodelay(int fd) {
    int one = 1;
    return setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) == 0;
}

bool write_all(int fd, const void * p, size_t n) {
    const char * c = (const char *) p;
    while (n) {
        const ssize_t w = ::write(fd, c, n);
        if (w <= 0) return false;
        c += w;
        n -= (size_t) w;
    }
    return true;
}

bool read_all(int fd, void * p, size_t n) {
    char * c = (char *) p;
    while (n) {
        const ssize_t r = ::read(fd, c, n);
        if (r <= 0) return false;
        c += r;
        n -= (size_t) r;
    }
    return true;
}

// A star: rank 0 collects every endpoint and hands the full table back. Small
// and synchronous, which is what a one-off exchange should be.
bool exchange(const FastReduce::Config & cfg, const Endpoint & mine,
              std::vector<Endpoint> & all, std::string * err) {
    all.assign((size_t) cfg.size, Endpoint{});
    all[(size_t) cfg.rank] = mine;
    if (cfg.size <= 1) return true;

    auto fail = [&](const char * what) {
        if (err) *err = std::string("fast-reduce bootstrap: ") + what + ": " + std::strerror(errno);
        return false;
    };

    if (cfg.rank == 0) {
        const int lst = ::socket(AF_INET, SOCK_STREAM, 0);
        if (lst < 0) return fail("socket");
        int one = 1;
        setsockopt(lst, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = INADDR_ANY;
        a.sin_port = htons((uint16_t) cfg.bootstrap_port);
        if (::bind(lst, (sockaddr *) &a, sizeof(a)) != 0) { ::close(lst); return fail("bind"); }
        if (::listen(lst, cfg.size) != 0) { ::close(lst); return fail("listen"); }

        std::vector<int> peers((size_t) cfg.size, -1);
        for (int i = 1; i < cfg.size; ++i) {
            const int fd = ::accept(lst, nullptr, nullptr);
            if (fd < 0) { ::close(lst); return fail("accept"); }
            set_nodelay(fd);
            int32_t r = -1;
            Endpoint e{};
            if (!read_all(fd, &r, sizeof(r)) || !read_all(fd, &e, sizeof(e)) ||
                r <= 0 || r >= cfg.size) {
                ::close(fd); ::close(lst);
                return fail("read endpoint");
            }
            all[(size_t) r] = e;
            peers[(size_t) r] = fd;
        }
        ::close(lst);
        for (int i = 1; i < cfg.size; ++i) {
            if (!write_all(peers[(size_t) i], all.data(), sizeof(Endpoint) * all.size())) {
                for (int fd : peers) if (fd >= 0) ::close(fd);
                return fail("write table");
            }
        }
        for (int fd : peers) if (fd >= 0) ::close(fd);
        return true;
    }

    // Workers retry: rank 0 may still be loading when they get here.
    int fd = -1;
    for (int attempt = 0; attempt < 600; ++attempt) {
        fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return fail("socket");
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons((uint16_t) cfg.bootstrap_port);
        if (::inet_pton(AF_INET, cfg.bootstrap_host.c_str(), &a.sin_addr) != 1) {
            ::close(fd);
            if (err) *err = "fast-reduce bootstrap: bad host " + cfg.bootstrap_host;
            return false;
        }
        if (::connect(fd, (sockaddr *) &a, sizeof(a)) == 0) break;
        ::close(fd);
        fd = -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (fd < 0) return fail("connect");
    set_nodelay(fd);
    const int32_t r = cfg.rank;
    if (!write_all(fd, &r, sizeof(r)) || !write_all(fd, &mine, sizeof(mine)) ||
        !read_all(fd, all.data(), sizeof(Endpoint) * all.size())) {
        ::close(fd);
        return fail("exchange");
    }
    ::close(fd);
    return true;
}

}  // namespace

struct FastReduceImpl {
    FastReduce::Config cfg;
    bool up = false;

    ibv_context * ctx = nullptr;
    ibv_pd *      pd = nullptr;
    ibv_cq *      cq = nullptr;
    ibv_mr *      mr_data = nullptr;
    ibv_mr *      mr_flag = nullptr;
    std::vector<ibv_qp *> qps;              // one per peer, null at own rank
    std::vector<Endpoint> peers;

    // Pinned, NIC-registered, GPU-mapped. One region for payloads and one for
    // flags, so a flag write cannot be reordered behind a payload write on a
    // different memory region.
    float *    data_host = nullptr;         // [slots][size][max_elems]
    float *    data_dev = nullptr;
    uint32_t * flag_host = nullptr;         // [slots][size] recv + [slots] pub + [slots] done
    uint32_t * flag_dev = nullptr;

    // Device-visible pointer tables, one row per slot.
    const float ** peer_bufs = nullptr;     // [slots][size-1]
    const float ** peer_bufs_dev = nullptr;
    uint32_t **    peer_flags = nullptr;    // [slots][size-1]
    uint32_t **    peer_flags_dev = nullptr;

    uint32_t * timed_out_host = nullptr;
    uint32_t * timed_out_dev = nullptr;

    std::vector<int> pending_n;             // payload length per slot, host side
    std::atomic<uint64_t> seq{0};
    std::atomic<bool> stop{false};
    std::thread progress;

    size_t slot_stride() const { return (size_t) cfg.max_elems * (size_t) cfg.size; }
    // My own row inside a slot is never written by a peer; peers write theirs.
    float * recv_row(int slot, int peer) {
        return data_host + (size_t) slot * slot_stride() + (size_t) peer * cfg.max_elems;
    }
    uint32_t * recv_flag(int slot, int peer) {
        return flag_host + (size_t) slot * (size_t) cfg.size + (size_t) peer;
    }
    uint32_t * pub_flag(int slot) {
        return flag_host + (size_t) cfg.slots * (size_t) cfg.size + (size_t) slot;
    }
    uint32_t * done_flag(int slot) {
        return flag_host + (size_t) cfg.slots * (size_t) cfg.size +
               (size_t) cfg.slots + (size_t) slot;
    }
    size_t flag_count() const {
        return (size_t) cfg.slots * (size_t) cfg.size + 2u * (size_t) cfg.slots;
    }
};

FastReduce::FastReduce() : p_(new FastReduceImpl()) {}
FastReduce::~FastReduce() { shutdown(); }

bool FastReduce::ok() const { return p_ && p_->up; }
uint64_t FastReduce::submitted() const { return p_ ? p_->seq.load() : 0; }
uint64_t FastReduce::timed_out() const {
    return (p_ && p_->timed_out_host) ? *p_->timed_out_host : 0;
}

bool FastReduce::init(const Config & cfg, std::string * err) {
    auto & s = *p_;
    s.cfg = cfg;
    if (cfg.size <= 1) { s.up = false; return true; }

    auto bail = [&](const std::string & what) {
        if (err) *err = "fast-reduce: " + what;
        shutdown();
        return false;
    };

    // ── device ────────────────────────────────────────────────────────────
    int n_dev = 0;
    ibv_device ** devs = ibv_get_device_list(&n_dev);
    if (!devs) return bail("ibv_get_device_list failed");
    ibv_device * dev = nullptr;
    for (int i = 0; i < n_dev; ++i) {
        if (cfg.hca.empty() || cfg.hca == ibv_get_device_name(devs[i])) {
            dev = devs[i];
            break;
        }
    }
    if (!dev) { ibv_free_device_list(devs); return bail("no HCA named " + cfg.hca); }
    s.ctx = ibv_open_device(dev);
    ibv_free_device_list(devs);
    if (!s.ctx) return bail("ibv_open_device failed");

    s.pd = ibv_alloc_pd(s.ctx);
    if (!s.pd) return bail("ibv_alloc_pd failed");
    // Two work requests per peer per reduction, a step's worth in flight.
    s.cq = ibv_create_cq(s.ctx, 4096, nullptr, nullptr, 0);
    if (!s.cq) return bail("ibv_create_cq failed");

    // ── buffers ───────────────────────────────────────────────────────────
    const size_t data_elems = (size_t) cfg.slots * s.slot_stride();
    if (hipHostMalloc((void **) &s.data_host, data_elems * sizeof(float),
                      hipHostMallocDefault) != hipSuccess) {
        return bail("hipHostMalloc for the payload region failed");
    }
    std::memset(s.data_host, 0, data_elems * sizeof(float));
    if (hipHostMalloc((void **) &s.flag_host, s.flag_count() * sizeof(uint32_t),
                      hipHostMallocDefault) != hipSuccess) {
        return bail("hipHostMalloc for the flag region failed");
    }
    std::memset(s.flag_host, 0, s.flag_count() * sizeof(uint32_t));
    if (hipHostMalloc((void **) &s.timed_out_host, sizeof(uint32_t) * 16,
                      hipHostMallocDefault) != hipSuccess) {
        return bail("hipHostMalloc for the timeout word failed");
    }
    *s.timed_out_host = 0;

    if (hipHostGetDevicePointer((void **) &s.data_dev, s.data_host, 0) != hipSuccess ||
        hipHostGetDevicePointer((void **) &s.flag_dev, s.flag_host, 0) != hipSuccess ||
        hipHostGetDevicePointer((void **) &s.timed_out_dev, s.timed_out_host, 0) != hipSuccess) {
        return bail("hipHostGetDevicePointer failed");
    }

    s.mr_data = ibv_reg_mr(s.pd, s.data_host, data_elems * sizeof(float),
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    s.mr_flag = ibv_reg_mr(s.pd, s.flag_host, s.flag_count() * sizeof(uint32_t),
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!s.mr_data || !s.mr_flag) return bail("ibv_reg_mr failed (memlock limit?)");

    // ── queue pairs ───────────────────────────────────────────────────────
    s.qps.assign((size_t) cfg.size, nullptr);
    for (int r = 0; r < cfg.size; ++r) {
        if (r == cfg.rank) continue;
        ibv_qp_init_attr ia{};
        ia.send_cq = s.cq;
        ia.recv_cq = s.cq;
        ia.qp_type = IBV_QPT_RC;
        ia.cap.max_send_wr = 4096;
        ia.cap.max_recv_wr = 1;
        ia.cap.max_send_sge = 1;
        ia.cap.max_recv_sge = 1;
        ia.cap.max_inline_data = 64;      // the flag write goes inline
        s.qps[(size_t) r] = ibv_create_qp(s.pd, &ia);
        if (!s.qps[(size_t) r]) return bail("ibv_create_qp failed");

        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_num = (uint8_t) cfg.ib_port;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(s.qps[(size_t) r], &a,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS) != 0) {
            return bail("ibv_modify_qp to INIT failed");
        }
    }

    // The memory keys and addresses are the same for every peer; only the
    // queue pair numbers differ.
    Endpoint mine{};
    mine.rkey = s.mr_data->rkey;
    mine.data_addr = (uint64_t) (uintptr_t) s.data_host;
    mine.flag_addr = (uint64_t) (uintptr_t) s.flag_host;
    mine.psn = 0;
    ibv_gid gid{};
    if (ibv_query_gid(s.ctx, (uint8_t) cfg.ib_port, cfg.gid_index, &gid) != 0) {
        return bail("ibv_query_gid failed");
    }
    std::memcpy(mine.gid, gid.raw, 16);

    if (cfg.size > kMaxRanks) return bail("more ranks than the endpoint carries");
    for (int r = 0; r < cfg.size; ++r) {
        if (r == cfg.rank) continue;
        mine.qpn[r] = s.qps[(size_t) r]->qp_num;
    }

    std::vector<Endpoint> table;
    if (!exchange(cfg, mine, table, err)) { shutdown(); return false; }
    s.peers = table;

    for (int r = 0; r < cfg.size; ++r) {
        if (r == cfg.rank) continue;
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_RTR;
        a.path_mtu = IBV_MTU_1024;
        a.dest_qp_num = s.peers[(size_t) r].qpn[cfg.rank];
        a.rq_psn = 0;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer = 12;
        a.ah_attr.is_global = 1;
        a.ah_attr.port_num = (uint8_t) cfg.ib_port;
        a.ah_attr.grh.hop_limit = 64;
        a.ah_attr.grh.sgid_index = (uint8_t) cfg.gid_index;
        std::memcpy(a.ah_attr.grh.dgid.raw, s.peers[(size_t) r].gid, 16);
        if (ibv_modify_qp(s.qps[(size_t) r], &a,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            return bail("ibv_modify_qp to RTR failed");
        }
        ibv_qp_attr b{};
        b.qp_state = IBV_QPS_RTS;
        b.timeout = 14;
        b.retry_cnt = 7;
        b.rnr_retry = 7;
        b.sq_psn = 0;
        b.max_rd_atomic = 1;
        if (ibv_modify_qp(s.qps[(size_t) r], &b,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            return bail("ibv_modify_qp to RTS failed");
        }
    }

    // ── pointer tables, one row per slot ──────────────────────────────────
    const int n_peers = cfg.size - 1;
    const size_t rows = (size_t) cfg.slots * (size_t) n_peers;
    if (hipHostMalloc((void **) &s.peer_bufs, rows * sizeof(const float *),
                      hipHostMallocDefault) != hipSuccess ||
        hipHostMalloc((void **) &s.peer_flags, rows * sizeof(uint32_t *),
                      hipHostMallocDefault) != hipSuccess) {
        return bail("hipHostMalloc for the pointer tables failed");
    }
    for (int slot = 0; slot < cfg.slots; ++slot) {
        int k = 0;
        for (int r = 0; r < cfg.size; ++r) {
            if (r == cfg.rank) continue;
            const size_t off = (size_t) slot * (size_t) n_peers + (size_t) k;
            s.peer_bufs[off] = s.data_dev +
                ((size_t) slot * s.slot_stride() + (size_t) r * cfg.max_elems);
            s.peer_flags[off] = s.flag_dev +
                ((size_t) slot * (size_t) cfg.size + (size_t) r);
            ++k;
        }
    }
    if (hipHostGetDevicePointer((void **) &s.peer_bufs_dev, s.peer_bufs, 0) != hipSuccess ||
        hipHostGetDevicePointer((void **) &s.peer_flags_dev, s.peer_flags, 0) != hipSuccess) {
        return bail("hipHostGetDevicePointer for the pointer tables failed");
    }

    s.pending_n.assign((size_t) cfg.slots, 0);

    // ── progress ──────────────────────────────────────────────────────────
    s.up = true;
    s.progress = std::thread([&s]() {
        const int n_peers_l = s.cfg.size - 1;
        uint64_t next = 1;
        std::vector<ibv_wc> wc(64);
        while (!s.stop.load(std::memory_order_relaxed)) {
            const int slot = (int) (next % (uint64_t) s.cfg.slots);
            volatile uint32_t * pub = s.pub_flag(slot);
            uint64_t spins = 0;
            while (*pub != (uint32_t) next) {
                if (s.stop.load(std::memory_order_relaxed)) return;
                if (++spins > 2000000ull) { spins = 0; std::this_thread::yield(); }
            }
            const int n = s.pending_n[(size_t) slot];
            for (int r = 0; r < s.cfg.size; ++r) {
                if (r == s.cfg.rank) continue;
                const Endpoint & pe = s.peers[(size_t) r];

                ibv_sge sge{};
                sge.addr = (uint64_t) (uintptr_t) (s.data_host +
                    (size_t) slot * s.slot_stride() + (size_t) s.cfg.rank * s.cfg.max_elems);
                sge.length = (uint32_t) ((size_t) n * sizeof(float));
                sge.lkey = s.mr_data->lkey;

                ibv_send_wr w{};
                w.wr_id = next;
                w.sg_list = &sge;
                w.num_sge = 1;
                w.opcode = IBV_WR_RDMA_WRITE;
                w.send_flags = 0;
                w.wr.rdma.remote_addr = pe.data_addr +
                    ((size_t) slot * s.slot_stride() + (size_t) s.cfg.rank * s.cfg.max_elems) * sizeof(float);
                w.wr.rdma.rkey = pe.rkey;

                // The flag rides the same queue pair, so reliable-connection
                // ordering puts it after the payload. That ordering is the
                // whole protocol: a flag that could overtake its data would
                // hand the peer a half-written buffer.
                const uint32_t flagv = (uint32_t) next;
                ibv_sge fsge{};
                fsge.addr = (uint64_t) (uintptr_t) &flagv;
                fsge.length = sizeof(uint32_t);
                fsge.lkey = 0;

                ibv_send_wr fw{};
                fw.wr_id = next;
                fw.sg_list = &fsge;
                fw.num_sge = 1;
                fw.opcode = IBV_WR_RDMA_WRITE;
                fw.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
                fw.wr.rdma.remote_addr = pe.flag_addr +
                    ((size_t) slot * (size_t) s.cfg.size + (size_t) s.cfg.rank) * sizeof(uint32_t);
                fw.wr.rdma.rkey = pe.rkey;

                w.next = &fw;
                ibv_send_wr * bad = nullptr;
                if (ibv_post_send(s.qps[(size_t) r], &w, &bad) != 0) {
                    s.stop.store(true);
                    return;
                }
            }
            // Drain completions so the send queue does not fill. A failed
            // write is not recoverable here -- the peer is already waiting on
            // a flag that will never arrive -- so it stops the thread and the
            // waiting kernels time out into a counted, visible failure.
            int done = 0;
            do {
                done = ibv_poll_cq(s.cq, (int) wc.size(), wc.data());
                for (int i = 0; i < done; ++i) {
                    if (wc[(size_t) i].status != IBV_WC_SUCCESS) {
                        std::fprintf(stderr,
                                     "[fast-reduce] write failed: %s (seq %llu)\n",
                                     ibv_wc_status_str(wc[(size_t) i].status),
                                     (unsigned long long) wc[(size_t) i].wr_id);
                        s.stop.store(true);
                        return;
                    }
                }
            } while (done > 0);
            ++next;
        }
    });

    return true;
}

bool FastReduce::submit(float * data, size_t n, void * stream) {
    auto & s = *p_;
    if (!s.up || n == 0 || n > (size_t) s.cfg.max_elems) return false;

    const uint64_t seq = s.seq.fetch_add(1, std::memory_order_relaxed) + 1;
    const int slot = (int) (seq % (uint64_t) s.cfg.slots);
    // The progress thread reads this once it sees the flag, and the flag is
    // raised by a kernel launched below -- so writing it here is ordered
    // ahead of any possible read.
    s.pending_n[(size_t) slot] = (int) n;

    const int n_peers = s.cfg.size - 1;
    float * staging = s.data_dev + (size_t) slot * s.slot_stride() +
                      (size_t) s.cfg.rank * s.cfg.max_elems;

    // A bound, not a deadline: about a second of spinning at this clock. A
    // rank that has died must surface as a wrong answer with a counter, not as
    // a GPU that never returns.
    constexpr uint64_t kSpin = 2000000000ull;

    fast_reduce_launch_publish(data, staging,
                               s.flag_dev + (size_t) s.cfg.slots * (size_t) s.cfg.size + (size_t) slot,
                               s.flag_dev + (size_t) s.cfg.slots * (size_t) s.cfg.size +
                                   (size_t) s.cfg.slots + (size_t) slot,
                               (uint32_t) seq, (uint32_t) (s.cfg.slots - 2), (int) n,
                               kSpin, (hipStream_t) stream);
    fast_reduce_launch_wait_add(data,
                                s.peer_bufs_dev + (size_t) slot * (size_t) n_peers,
                                s.peer_flags_dev + (size_t) slot * (size_t) n_peers,
                                s.flag_dev + (size_t) s.cfg.slots * (size_t) s.cfg.size +
                                    (size_t) s.cfg.slots + (size_t) slot,
                                (uint32_t) seq, n_peers, (int) n, kSpin,
                                s.timed_out_dev, (hipStream_t) stream);
    return true;
}

void FastReduce::shutdown() {
    if (!p_) return;
    auto & s = *p_;
    s.stop.store(true);
    if (s.progress.joinable()) s.progress.join();
    s.up = false;

    for (ibv_qp * q : s.qps) if (q) ibv_destroy_qp(q);
    s.qps.clear();
    if (s.mr_data) { ibv_dereg_mr(s.mr_data); s.mr_data = nullptr; }
    if (s.mr_flag) { ibv_dereg_mr(s.mr_flag); s.mr_flag = nullptr; }
    if (s.cq) { ibv_destroy_cq(s.cq); s.cq = nullptr; }
    if (s.pd) { ibv_dealloc_pd(s.pd); s.pd = nullptr; }
    if (s.ctx) { ibv_close_device(s.ctx); s.ctx = nullptr; }

    if (s.peer_bufs) { hipHostFree(s.peer_bufs); s.peer_bufs = nullptr; }
    if (s.peer_flags) { hipHostFree(s.peer_flags); s.peer_flags = nullptr; }
    if (s.data_host) { hipHostFree(s.data_host); s.data_host = nullptr; }
    if (s.flag_host) { hipHostFree(s.flag_host); s.flag_host = nullptr; }
    if (s.timed_out_host) { hipHostFree(s.timed_out_host); s.timed_out_host = nullptr; }
}

}  // namespace dflash::cluster

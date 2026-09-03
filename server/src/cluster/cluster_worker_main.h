// cluster_worker_main.h - process entry point for ranks 1..N-1.
//
// server_main.cpp parses the full command line as usual, then, when
// --cluster-rank is greater than zero, hands BackendArgs and the feature
// config here instead of starting the HTTP server. The worker builds the
// same DeepSeek4 backend the head builds, joins the cluster over the control
// channel, and then replays every RequestMsg / BackendOpMsg the head sends
// until Shutdown (return 0) or Abort / failure (return non-zero, so a
// supervisor restarts the whole cluster; M1 has no re-init).

#pragma once

#include "common/backend_args.h"

namespace dflash::cluster {

int run_cluster_worker(common::BackendArgs & args,
                       const common::BackendFeatureConfig & features,
                       int argc, char ** argv);

}  // namespace dflash::cluster

# source/peract 相对上游的完整改动（自动导出，勿手改）
#
# 为什么有这个文件：
#   source/peract 是独立 git 仓库，remote 指向上游 https://github.com/peract/peract.git，
#   我们没有写权限 —— 本项目对 PerAct 的全部改动只存在于本机的本地提交里，
#   一旦这台共享机器出问题就全没了。这里把 delta 导出到主仓库，随主仓库推到远程。
#
# 恢复方式：
#   cd source/peract && git checkout origin/main && git apply ../../source_patches/peract.patch
#
# 本机的本地提交（git log origin/main..HEAD）：
#   f53578c P8-1: B4 臂接入 —— 臂能力不再写死，注入层版本以臂为唯一真源
#   c677dfd P5-2: 修 fill_replay 的两处集成缺陷
#   5429304 P5-1: 修复 Stage 3 集成的 6 个 bug
#   d923f28 P4-5: 冻结码本随 agent 上设备
#   9bdacc6 P4-3/4: replay 子任务字段 + cache 接入 + 按臂装配
#   369cc64 P4-2: VQAP 码向量注入点 + Stage 3 冻结边界
#   700da02 env: adapt to python3.10 / torch2.4 / numpy1.26 for RTX 5880 Ada (sm_89)
#
# 导出时间：2026-09-07 03:02:04
# 涉及文件：
#    agents/arm/launch_utils.py                     |   2 +-
#    agents/baselines/bc_lang/launch_utils.py       |   2 +-
#    agents/baselines/vit_bc_lang/launch_utils.py   |   2 +-
#    agents/c2farm_lingunet_bc/launch_utils.py      |   2 +-
#    agents/peract_bc/launch_utils.py               | 173 +++++++++++++++++-
#    agents/peract_bc/perceiver_lang_io.py          |  17 +-
#    agents/peract_bc/qattention_peract_bc_agent.py | 239 ++++++++++++++++++++++---
#    conf/eval.yaml                                 |  19 +-
#    conf/stage3.yaml                               | 100 +++++++++++
#    eval.py                                        |  42 +++++
#    helpers/custom_rlbench_env.py                  |  24 +++
#    requirements.txt                               |  14 +-
#    run_seed_fn.py                                 | 102 +++++++++--
#    train.py                                       |  33 +++-
#    voxel/voxel_grid.py                            |  24 +--
#    15 files changed, 729 insertions(+), 66 deletions(-)

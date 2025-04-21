#!/bin/bash
export SWING_TESTBED_PATH=~/swing_testbed
export SWING_REPOS_DIR_PATH=~/swing_repos
export SWING_INDEXES_PATH=/home/xiongjing/resource_dir/SwingBench/tmpdata/indexes #~/swing_indexes
export CI_TOOL_NAME=act

python swebench/harness/agent_battle.py --workdir $SWING_TESTBED_PATH --src_folder $SWING_REPOS_DIR_PATH --retriever_index_dir $SWING_INDEXES_PATH --ci_tool_name $CI_TOOL_NAME
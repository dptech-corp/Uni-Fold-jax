#!/bin/bash

# Copyright 2021 Beijing DP Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Searching for MSA with MMSeqs2.

set -e

MMSEQS="${1}"
QUERY="${2}"
DBBASE="${3}"
BASE="${4}"
DB1="${5}"
DB2="${6}"
DB3="${7}"
USE_ENV="${8}"
USE_TEMPLATES="${9}"
FILTER="${10}"
EXPAND_EVAL=inf
ALIGN_EVAL=10
DIFF=3000
QSC=-20.0
MAX_ACCEPT=1000000
if [[ "${FILTER}" == "1" ]]; then
  # 0.1 was not used in benchmarks due to POSIX shell bug in line above
  #   EXPAND_EVAL=0.1
  ALIGN_EVAL=10
  QSC=0.8
  MAX_ACCEPT=100000
fi
export MMSEQS_CALL_DEPTH=1

SEARCH_PARAM=("--num-iterations" "3" "--db-load-mode" "2" "-a" "-s" "8" "-e" "0.1" "--max-seqs" "10000")
FILTER_PARAM=("--filter-msa" "${FILTER}" "--filter-min-enable" "1000" "--diff" "${DIFF}" "--qid" "0.0,0.2,0.4,0.6,0.8,1.0" "--qsc" "0" "--max-seq-id" "0.95")
EXPAND_PARAM=("--expansion-mode" "0" "-e" "${EXPAND_EVAL}" "--expand-filter-clusters" "${FILTER}" "--max-seq-id" "0.95")

mkdir -p "${BASE}"

"${MMSEQS}" createdb "${QUERY}" "${BASE}/qdb"
"${MMSEQS}" search "${BASE}/qdb" "${DBBASE}/${DB1}" "${BASE}/res" "${BASE}/tmp" "${SEARCH_PARAM[@]}"
"${MMSEQS}" expandaln "${BASE}/qdb" "${DBBASE}/${DB1}.idx" "${BASE}/res" "${DBBASE}/${DB1}.idx" "${BASE}/res_exp" --db-load-mode 2 "${EXPAND_PARAM[@]}"
"${MMSEQS}" mvdb "${BASE}/tmp/latest/profile_1" "${BASE}/prof_res"
"${MMSEQS}" lndb "${BASE}/qdb_h" "${BASE}/prof_res_h"
"${MMSEQS}" align "${BASE}/prof_res" "${DBBASE}/${DB1}.idx" "${BASE}/res_exp" "${BASE}/res_exp_realign" --db-load-mode 2 -e "${ALIGN_EVAL}" --max-accept "${MAX_ACCEPT}" --alt-ali 10 -a
"${MMSEQS}" filterresult "${BASE}/qdb" "${DBBASE}/${DB1}.idx" "${BASE}/res_exp_realign" "${BASE}/res_exp_realign_filter" --db-load-mode 2 --qid 0 --qsc $QSC --diff 0 --max-seq-id 1.0 --filter-min-enable 100
"${MMSEQS}" result2msa "${BASE}/qdb" "${DBBASE}/${DB1}.idx" "${BASE}/res_exp_realign_filter" "${BASE}/uniref.a3m" --msa-format-mode 6 --db-load-mode 2 "${FILTER_PARAM[@]}"
"${MMSEQS}" rmdb "${BASE}/res_exp_realign"
"${MMSEQS}" rmdb "${BASE}/res_exp"
"${MMSEQS}" rmdb "${BASE}/res"
"${MMSEQS}" rmdb "${BASE}/res_exp_realign_filter"

if [[ "${USE_TEMPLATES}" == "1" ]]; then
  "${MMSEQS}" search "${BASE}/prof_res" "${DBBASE}/${DB2}" "${BASE}/res_pdb" "${BASE}/tmp" --db-load-mode 2 -s 7.5 -a -e 0.1
  "${MMSEQS}" convertalis "${BASE}/prof_res" "${DBBASE}/${DB2}.idx" "${BASE}/res_pdb" "${BASE}/${DB2}.m8" --format-output query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,cigar --db-load-mode 2
  "${MMSEQS}" rmdb "${BASE}/res_pdb"
fi

if [[ "${USE_ENV}" == "1" ]]; then
  "${MMSEQS}" search "${BASE}/prof_res" "${DBBASE}/${DB3}" "${BASE}/res_env" "${BASE}/tmp" "${SEARCH_PARAM[@]}"
  "${MMSEQS}" expandaln "${BASE}/prof_res" "${DBBASE}/${DB3}.idx" "${BASE}/res_env" "${DBBASE}/${DB3}.idx" "${BASE}/res_env_exp" -e "${EXPAND_EVAL}" --expansion-mode 0 --db-load-mode 2
  "${MMSEQS}" align "${BASE}/tmp/latest/profile_1" "${DBBASE}/${DB3}.idx" "${BASE}/res_env_exp" "${BASE}/res_env_exp_realign" --db-load-mode 2 -e "${ALIGN_EVAL}" --max-accept "${MAX_ACCEPT}" --alt-ali 10 -a
  "${MMSEQS}" filterresult "${BASE}/qdb" "${DBBASE}/${DB3}.idx" "${BASE}/res_env_exp_realign" "${BASE}/res_env_exp_realign_filter" --db-load-mode 2 --qid 0 --qsc $QSC --diff 0 --max-seq-id 1.0 --filter-min-enable 100
  "${MMSEQS}" result2msa "${BASE}/qdb" "${DBBASE}/${DB3}.idx" "${BASE}/res_env_exp_realign_filter" "${BASE}/bfd.mgnify30.metaeuk30.smag30.a3m" --msa-format-mode 6 --db-load-mode 2 "${FILTER_PARAM[@]}"
  "${MMSEQS}" rmdb "${BASE}/res_env_exp_realign_filter"
  "${MMSEQS}" rmdb "${BASE}/res_env_exp_realign"
  "${MMSEQS}" rmdb "${BASE}/res_env_exp"
  "${MMSEQS}" rmdb "${BASE}/res_env"
fi
"${MMSEQS}" rmdb "${BASE}/qdb"
"${MMSEQS}" rmdb "${BASE}/qdb_h"
"${MMSEQS}" rmdb "${BASE}/res"

rm -f -- "${BASE}/prof_res"*
rm -rf -- "${BASE}/tmp"

inductor_skip_list=(
  "test_check_accuracy.py"
  "test_debug_msg.py"
  "test_embedding.py"
  "test_force_fallback.py"
  "test_foreach_add.py"
  "test_geometric.py"
  "test_lazy_register.py"
  "test_npu_dtype_cast.py"
  )

TEST_inductor="${WORKSPACE}/ascend/examples/inductor_cases"
# remove inductor cache
rm -rf /tmp/torchinductor_*
cd ${TEST_inductor}
git init
git remote add origin http://gitee.com/ascend/pytorch.git
git config core.sparsecheckout true
echo "test/_inductor" >> .git/info/sparse-checkout
git pull origin v2.6.0:master
TEST_inductor_cases_path="${TEST_inductor}/test/_inductor"
cd ${TEST_inductor_cases_path}
export PYTHONPATH="${PYTHONPATH}:${TEST_inductor_cases_path}"
for skip_case in ${inductor_skip_list[@]};
do
  if [ -e "${TEST_inductor_cases_path}/${skip_case}" ];then
    echo "skip test case of ${skip_case}"
    mv ${skip_case} "${skip_case}_skip"
  fi
done

INDUCTOR_CASE_LOG_FILE="${WORKSPACE}/test_inductor_case_$(date +%Y%m%d).log"
pytest -n 16 --dist=load . 2>&1 | tee -a "$INDUCTOR_CASE_LOG_FILE"

cp "$INDUCTOR_CASE_LOG_FILE" "/home/daily_log"
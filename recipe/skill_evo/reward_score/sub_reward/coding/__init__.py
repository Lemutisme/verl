from . import block_level_process_reward
from . import compiler_runtime_feedback
from . import executed_token_credit
from . import static_analysis_reward
from . import unit_test_pass_rate


MODULES = {
    "unit_test_pass_rate": unit_test_pass_rate,
    "compiler_runtime_feedback": compiler_runtime_feedback,
    "static_analysis_reward": static_analysis_reward,
    "executed_token_credit": executed_token_credit,
    "block_level_process_reward": block_level_process_reward,
}

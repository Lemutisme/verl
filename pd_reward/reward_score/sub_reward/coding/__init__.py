from . import block_level_process_reward
from . import code_extractability_reward
from . import compiler_runtime_feedback
from . import executed_token_credit
from . import syntax_validity_reward
from . import static_analysis_reward
from . import unit_test_pass_rate


MODULES = {
    "code_extractability_reward": code_extractability_reward,
    "syntax_validity_reward": syntax_validity_reward,
    "unit_test_pass_rate": unit_test_pass_rate,
    "compiler_runtime_feedback": compiler_runtime_feedback,
    "static_analysis_reward": static_analysis_reward,
    "executed_token_credit": executed_token_credit,
    "block_level_process_reward": block_level_process_reward,
}

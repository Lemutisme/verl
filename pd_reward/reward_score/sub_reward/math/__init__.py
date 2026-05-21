from . import answer_efficiency_reward
from . import answer_extractability_reward
from . import consistency_reward
from . import executable_unit_pass_rate_reward
from . import final_answer_reward
from . import prefix_consistency_reward
from . import step_arithmetic_validity_reward
from . import trace_efficiency_reward


MODULES = {
    "final_answer_reward": final_answer_reward,
    "answer_efficiency_reward": answer_efficiency_reward,
    "consistency_reward": consistency_reward,
    "executable_unit_pass_rate_reward": executable_unit_pass_rate_reward,
    "step_arithmetic_validity_reward": step_arithmetic_validity_reward,
    "prefix_consistency_reward": prefix_consistency_reward,
    "trace_efficiency_reward": trace_efficiency_reward,
    "answer_extractability_reward": answer_extractability_reward,
}

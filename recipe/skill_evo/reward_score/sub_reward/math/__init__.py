from . import boxed_format_reward
from . import final_answer_reward
from . import reasoning_quality_heuristic
from . import progress_reward
from . import reasoning_distance_reward
from . import step_reachability_reward


MODULES = {
    "final_answer_reward": final_answer_reward,
    "boxed_format_reward": boxed_format_reward,
    "step_reachability_reward": step_reachability_reward,
    "progress_reward": progress_reward,
    "reasoning_quality_heuristic": reasoning_quality_heuristic,
    "reasoning_distance_reward": reasoning_distance_reward,
}

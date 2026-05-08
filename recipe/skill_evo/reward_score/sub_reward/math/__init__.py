from . import answer_efficiency_reward
from . import consistency_reward
from . import final_answer_reward


MODULES = {
    "final_answer_reward": final_answer_reward,
    "answer_efficiency_reward": answer_efficiency_reward,
    "consistency_reward": consistency_reward,
}

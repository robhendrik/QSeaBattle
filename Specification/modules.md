Modules

Core game & logging
* Game infrastructure
  * game_layout. py
    *	class GameLayout
  * game_env. py
    * class GameEnv
  * game. py
    * class Game
  * tournament. py
    * class Tournament
  * tournament_log. py
    * class TournamentLog

Players
* Non-trainable, non-assisted players
  * players_base. py
    * class Players
    * class PlayerA
    * class PlayerB
  * simple_player_a. py
    * class SimplePlayerA
  * simple_player_b. py
    * class SimplePlayerB
  * simple_players. py
    * class SimplePlayers
  * majority_player_a. py
    * class MajorityPlayerA
  * majority_player_b. py
    * class MajorityPlayerB
  * majority_players. py
    * class MajorityPlayers
* Trainable non-assisted players
  * neural_net_player_a. py
    * class NeuralNetPlayerA
  * neural_net_player_b. py
    * class NeuralNetPlayerB
  * neural_net_players. py
    * class NeuralNetPlayers
* Non-trainable, assisted players
  * assisted_player_a. py
    * class AssistedPlayerA
            (deprecated, refer to PRAssistedPlayerA)
  * assisted_player_b. py
    * class AssistedPlayerB
            (deprecated, refer to PRAssistedPlayerB)
  * assisted_players. py
    * class AssistedPlayers
            (deprecated, refer to PRAssistedPlayers)
  * pr_assisted_player_a. py
    * class PRAssistedPlayerA
  * pr_assisted_player_b. py
    * class PRAssistedPlayerB
  * pr_assisted_players. py
    * class PRAssistedPlayers

Assistance resources
* Non-trainable
  * shared_randomness. py
    * class SharedRandomness
        (deprecated, refer to PRAssisted)
* Non-trainable
  * pr_assisted.py
    * class PRAssisted

Utilities
* reference_performance_utilities.py
* logit_utils.py
* dru_utils.py

---
Shared randomness
•	shared_randomness_layer. py
o	class SharedRandomnessLayer
Linear trainable assisted model A (with shared randomness)
•	lin_measurement_layer_a. py
o	class LinMeasurementLayerA
•	lin_combine_layer_a. py
o	class LinCombineLayerA
•	lin_trainable_assisted_model_a. py
o	class LinTrainableAssistedModelA
Linear trainable assisted model B (with shared randomness)
•	lin_measurement_layer_b. py
o	class LinMeasurementLayerB
•	lin_combine_layer_b. py
o	class LinCombineLayerB
•	lin_trainable_assisted_model_b. py
o	class LinTrainableAssistedModelB
Pyramid trainable assisted model A (with shared randomness)
•	pyr_measurement_layer_a. py
o	class PyrMeasurementLayerA
•	pyr_combine_layer_a. py
o	class PyrCombineLayerA
•	pyr_trainable_assisted_model_a. py
o	class PyrTrainableAssistedModelA
Pyramid trainable assisted model B (with shared randomness)
•	pyr_measurement_layer_b. py
o	class PyrMeasurementLayerB
•	pyr_combine_layer_b. py
o	class PyrCombineLayerB
•	pyr_trainable_assisted_model_b. py
o	class PyrTrainableAssistedModelB

Combined trainable assisted players
•	trainable_assisted_players. py
o	class TrainableAssistedPlayers
•	trainable_assisted_player_b. py
o	class TrainableAssistedPlayerB
•	trainable_assisted_player_a. py
o	class TrainableAssistedPlayerA


Utilities for imitation training neural net players
•	lin_trainable_assisted_imitation_utils.py

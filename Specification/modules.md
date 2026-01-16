Modules

Core game & logging
* Game infrastructure
  * game_layout.py
    *	class GameLayout
  * game_env.py
    * class GameEnv
  * game.py
    * class Game
  * tournament.py
    * class Tournament
  * tournament_log.py
    * class TournamentLog

Players
* Non-trainable, non-assisted players
  * players_base.py
    * class Players
    * class PlayerA
    * class PlayerB
  * simple_player_a.py
    * class SimplePlayerA
  * simple_player_b.py
    * class SimplePlayerB
  * simple_players.py
    * class SimplePlayers
  * majority_player_a.py
    * class MajorityPlayerA
  * majority_player_b.py
    * class MajorityPlayerB
  * majority_players.py
    * class MajorityPlayers
* Trainable non-assisted players
  * neural_net_player_a.py
    * class NeuralNetPlayerA
  * neural_net_player_b.py
    * class NeuralNetPlayerB
  * neural_net_players.py
    * class NeuralNetPlayers
* Non-trainable, assisted players
  * assisted_player_a.py
    * class AssistedPlayerA
            (deprecated, refer to PRAssistedPlayerA)
  * assisted_player_b.py
    * class AssistedPlayerB
            (deprecated, refer to PRAssistedPlayerB)
  * assisted_players.py
    * class AssistedPlayers
            (deprecated, refer to PRAssistedPlayers)
  * pr_assisted_player_a.py
    * class PRAssistedPlayerA
  * pr_assisted_player_b.py
    * class PRAssistedPlayerB
  * pr_assisted_players.py
    * class PRAssistedPlayers

Trainable, assisted players
  * trainable_assisted_players
    * TrainableAssistedPlayers
  * trainable_assisted_player_b
    * TrainableAssistedPlayerB
  * trainable_assisted_player_a
    * TrainableAssistedPlayerA

Assistance resources
* Non-trainable
  * shared_randomness.py
    * class SharedRandomness
        (deprecated, refer to PRAssisted)
* Non-trainable
  * pr_assisted.py
    * class PRAssisted
* Trainable
  * shared_randomness_layer.py
    * class SharedRandomnessLayer
        (deprecated, refer to PRAssistedLayer)
* Trainable
  * pr_assisted_layer.py
    * class PRAssistedLayer

Models
Linear trainable assisted model A 
* lin_measurement_layer_a.py
  * LinMeasurementLayerA
* lin_combine_layer_a.py
  * LinCombineLayerA
* lin_trainable_assisted_model_a.py
  * LinTrainableAssistedModelA

Linear trainable assisted model B 
* lin_measurement_layer_b.py
  * class LinMeasurementLayerB
* lin_combine_layer_b.py
  * class LinCombineLayerB
* lin_trainable_assisted_model_b.py
  * class LinTrainableAssistedModelB

Pyramid trainable assisted model A 
* pyr_measurement_layer_a.py
  * PyrMeasurementLayerA
* pyr_combine_layer_a.py
  * PyrCombineLayerA
* pyr_trainable_assisted_model_a.py
  * PyrTrainableAssistedModelA

Pyramid trainable assisted model B 
* pyr_measurement_layer_b.py
  * class PyrMeasurementLayerB
* pyr_combine_layer_b.py
  * class PyrCombineLayerB
* pyr_trainable_assisted_model_b.py
  * class PyrTrainableAssistedModelB

Utilities
* reference_performance_utilities.py
* logit_utils.py
* dru_utils.py

---
Shared randomness

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



Utilities for imitation training neural net players
•	lin_trainable_assisted_imitation_utils.py


We have changed the name of a modules and a class. The end-state is that this is deprecated:
* Trainable
  * shared_randomness_layer. py
    * class SharedRandomnessLayer
and the new name is
* Trainable
  * pr_assisted_layer. py
    * class PRAssistedLayer
It means that SharedRandomnessLayer will become PRAssistedLayer. In general we remove the reference to shared randomness and following the naming in attached markdown (New naming for shared randomness.md). Attached you find this module pr_assisted_layer.py. 

If want make lin_model and the lower lower lin_measure and lin_combine modules use the new PRAssistedLayer. 

Check if the new naming leads to any change in docstrings or comments. Check is any function name or attribute or variable should be renamed. Then check if there is any reference to the old layer module or class that should be changed. List the changes for me and give me the downloadable modules for lin* with the changes implemented. A user should not see in naming, docstring, comments any confusing reference to shared randomness, we only havse share resources or which pr_assisted/PRAssisted is teh type we use here. 




First create new modules with PR, then in a second prompt we write the .md documentation and then in step 3 we change the orginal assisted to still exist, but refer to the new modules with a user warning (.... will be deprated, use ....) finally in step 4 we write new tests for the new name. So, everything sould continue working with backward compatibility and i get a warning is a notebook or script still uses the old interface. Can you implement step 3: change the orginal assisted to still exist, but refer to the new modules with a user warning (.... will be deprated, use ....). This should continue to work without changing downstream notebooks or scripts, jus the warning is added and the actual execution is move to the new module!




SYSTEM You are a technical writer who produces specification-grade, human-readable Markdown for MkDocs (Material). Output must be VALID MARKDOWN ONLY. CONSTRAINTS 
- Follow the exact style rules below and in the project STYLE.md (no deviations). - No HTML, no code in tables, no tabs/accordions. 
- Headings start at H1 and are hierarchical. 
- Use admonitions: note/warning/example/tip. 
- Include shapes and types for every argument and return value. 
- Include Preconditions, Postconditions, Errors. 
- Do NOT invent APIs. If an item is missing in code, but present in the design doc, include it under “Planned (design-spec)” with a short note. 
- If code and design disagree, add a “Deviations” section describing both sides without resolving the conflict. STYLE RULES (must adhere) 
- Arrays: e.g., “np.ndarray, dtype int {0,1}, shape (n2,)”. 
- Tensors: e.g., “tf.Tensor, dtype float32, shape (B, n2)”. 
- Symbols: use field_size, comms_size, n2, m consistently. 
- Keep line length under ~120 chars. 
- All rules as defined in attached STYLE.MD, unless the provided template overrules the rules in this style document. 

INPUTS 
* can mean anything in ['reference_performance', 'neural_net_imitation', 'dru', 'logits', 'pyr_trainable_assisted_imitation', 'lin_trainable_assisted_imitation']
1) Design document: QSeaBattleDesignDocument.docx 
2) Python module: *_utilities.py
3) Target: “module” page (see attached module_template.md) 
4) Output path hint: docs/utilities/*_utilities.md
5) Title: Module *_utilities 
6) Module import path (if known): 
Q_Sea_Battle.*_utilities



TASK Produce ONE Markdown pages “Module page template” (for a module) as shown below for each version of *. The page must be self-contained and ready for MkDocs. 

TEMPLATE TO USE module_template.md 

ADDITIONAL INSTRUCTIONS 
- “Examples” must be minimal, runnable pseudo-usage aligned with actual signatures. 
- Add “Testing Hooks” (suggested invariants) on module pages. 
- Add “Notes for Contributors” on class pages. 
- If shapes depend on GameLayout, state the derived constraints (m | n2, power-of-two if required). 
- End the page with a short “Changelog” initialized with today’s date and name 

Author: Rob Hendriks OUTPUT Return ONLY the Markdown body for the one page per * as a downloadable ZIP. No explanations, no preambles. Avoid this error in the generated MKDOCS from the Markdown files: fileciteturn3file0



We change names:
pyr_layers_step1.py → pyr_teacher_layers.py
pyr_models_step2.py → pyr_trainable_models.py

We added
lin_teacher_layers.py
lin_trainable_models.py

We deprecated
deprecated:
shared_randomness_layer.py/class SharedRandomnessLayer
and the new name is
pr_assisted_layer.py/class PRAssistedLayer

We change names:
pyr_layers_step1.py → pyr_teacher_layers.py
pyr_models_step2.py → pyr_trainable_models.py

We added
lin_teacher_layers.py
lin_trainable_models.py

We change for *_utils.py the module name to *_utilities.py

We redefined the abbreviation 'sr' to mean shared resources (this is wat sr stands for in eg sr_mode, PRAssisted is a type of shared resources, we do not mention shared randomness anymore, also not to refer that it has changed). 

Can you check this notebook and implement the change? Can you also us teh container lin_trainable_models and lin_teacher_layers?
pandoc -f markdown-yaml_metadata_block `
  docs/index.md `
  docs/algorithms.md `
  docs/conventions.md `
  docs/game/GameLayout.md `
  docs/game/GameEnv.md `
  docs/game/Game.md `
  docs/game/Tournament.md `
  docs/players/base.md `
  docs/players/deterministic.md `
  docs/players/assisted.md `
  docs/players/neural.md `
  docs/shared_randomness/shared_randomness.md `
  docs/shared_randomness/layer.md `
  docs/models/lin.md `
  docs/models/pyr.md `
  docs/training/imitation.md `
  docs/training/dru.md `
  docs/training/rl.md `
  docs/training/pyr_training.md `
  docs/utils/data_generation.md `
  docs/utils/imitation_training.md `
  docs/utils/layer_transfer.md `
  docs/utils/evaluation.md `
  docs/invariants.md `
  --toc `
  --number-sections `
  --pdf-engine=lualatex `
  -V mainfont="Consolas" `
  -V monofont="Consolas" `
  -V microtype=false `
  -o QSeaBattle_Spec_v0.1.pdf


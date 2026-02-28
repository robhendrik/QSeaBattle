$mdFiles = Get-ChildItem -Path docs -Filter *.md -Recurse | ForEach-Object { $_.FullName -replace '\\', '/' } | Sort-Object

pandoc -f markdown-yaml_metadata_block `
  $mdFiles `
  --toc `
  --number-sections `
  --pdf-engine=xelatex `
  -V geometry:margin=1in `
  --listings `
  -o QSeaBattle_Spec_v0.1.pdf


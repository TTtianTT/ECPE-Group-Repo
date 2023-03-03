在GPT3.5的三个模型，GPT3的6个模型上评估了性能（funsd数据集，zero-shot）



| Name               | Max Length |   Parameters |     f1 micro |    f1 macro| Fill Count |
|--------------------|------------|--------------|--------------|------------|------------|
| code-davinci-002   | 8k         | 175B         | 0.731        | 0.66       | 0          |
| text-davinci-002   | 4k         | 175B         | 0.639        | 0.45       | 54         |
| text-davinci-003   | 4k         | 175B         | 0.746        | 0.68       | 0          |
|                    |            |              |              |            |            |
| text-ada-001       | 2k         | 350M         | 0.480        | 0.16       | 44         |
| text-babbage-001   | 2k         | 1.3B         | 0.153        | 0.12       | 237        |
| text-curie-001     | 2k         | 6.7B         | 0.093        | 0.09       | 0          |
| text-davinci-001   | 2k         | 175B         | 0.313        | 0.32       | 0          |
| code-davinci-001   | 4k         | 175B         | 0.648        | 0.49       | 0          |
| code-cushman-001   | 2k         | 12B          | 0.696        | 0.60       | 0          |

# mt_luxembourgish

## Problematics

- 低数据源的卢森堡语的语料很少，导致训练数据不足。卢森堡语转英语比较准确，但是反向困难。
- 伪翻译的数据源不能够得到很好的翻译结果，但是可以作为训练数据源。
- 如何在单边语料以及仅有字典的情况下，如何达到更好的双向翻译性能。


## Methods

1. 伪翻译增强
2. 相关语料 类似于 德语或者法语增强。
3. 对于GPT4 进行知识蒸馏，提取更加多的语料数据，（不同源头: Meta， OPENAI， others）
4. 利用chatGPT模型的编造特点做数据增强，结合静态工作做cross check
5. FT 的模型选择，训练方法选择， 以及寻找反向的方法
6. 验证我们的数据方法可以在类似的语言 比如说挪威语和冰岛语之间做类似的工作，得到相同的结果

## Steps

1. 数据清洗。
2. 测试llama 3.2法语性能
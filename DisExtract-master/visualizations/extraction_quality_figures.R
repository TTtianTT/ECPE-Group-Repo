library(tidyverse)
library(ggthemes)
y = c(1765, 2625, 1539, 1107, 622, 217, 131, 156, 28, 0, 0)
x = c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ggplot(NULL, aes(x=x, y=y)) +
  geom_bar(stat="identity") +
  ylab('Number of examples') +
  xlab('Normalized Levenshtein distance') +
  theme(axis.text.y = element_text(colour="black")) +
  theme(axis.text.x = element_text(colour="black")) +
  theme_few(18)
ggsave("distance-distributions.pdf", width=8, height=6)

y = c(0.15395683, 0.24175824, 0.25625   , 0.26356589, 0.27126437,
          0.30872483, 0.31632319, 0.41912293, 0.41841004, 0.43892751,
          0.44396552, 0.45833333, 0.54144241, 0.69267364, 0.78498294)
x = levels=c('because','so','then','before',
             'after',
             'while',
             'but',
             'also',
             'although',
             'if',
             'though',
             'when',
             'as',
             'and',
             'still')
x = factor(x, levels=x)
ggplot(NULL, aes(x=x, y=1-y)) +
  geom_bar(stat="identity") +
  theme_few(18) +
  xlab("") +
  ylab("Extraction precision") +
  theme(axis.text.y = element_text(colour="black")) +
  theme(axis.text.x = element_text(colour="black")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ylim(0,1)
ggsave("extraction-error-rate.pdf", width=8, height=6)
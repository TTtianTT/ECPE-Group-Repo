source("startup.R")

unbalanced_filename = "/Users/erindb/Projects/DisExtract/model/exp/confusion_test.csv"
balanced_filename = "/Users/erindb/Projects/DisExtract/model/exp/perfectly_balanced_all_english.csv"
# input_filename = "/Users/erindb/Projects/DisExtract/model/exp/mostly_balanced_all_english.csv"

unbalanced_df = read.csv(unbalanced_filename) %>%
  mutate(labels = num(labels),
         preds = num(preds))
print(paste("unbalanced accuracy:", mean(unbalanced_df$labels==unbalanced_df$preds)))
# print(paste("mean:",
#             with(df %>% group_by(labels) %>% 
#                    summarise(accuracy = mean(labels==preds)),
#                  mean(accuracy))))

balanced_df = read.csv(balanced_filename) %>%
  mutate(labels = num(labels),
         preds = num(preds))
print(paste("balanced accuracy:", mean(balanced_df$labels==balanced_df$preds)))

class_labels = c(
    "after", "also", "although", "and", "as",
    "because", "before", "but", #"for example",
    #"however", 
    "if", "so", "still",
    "then", "though", "when", "while")

unbalanced_df = unbalanced_df %>%
  mutate(labels = factor(labels, levels=0:(length(class_labels)-1), labels=class_labels),
         preds = factor(preds, levels=0:(length(class_labels)-1), labels=class_labels))

balanced_df = balanced_df %>%
  mutate(labels = factor(labels, levels=0:(length(class_labels)-1), labels=class_labels),
         preds = factor(preds, levels=0:(length(class_labels)-1), labels=class_labels))

unbalanced_confusions = table(unbalanced_df) %>% as.data.frame() %>%
  mutate(
    labels = char(labels),
    preds = char(preds)
  ) %>%
  group_by(labels) %>%
  mutate(prop_gold = Freq / sum(Freq)) %>%
  ungroup() %>%
  group_by(preds) %>%
  mutate(prop_classifications = Freq / sum(Freq)) %>%
  ungroup()
unbalanced_diagonal = confusions %>% filter(labels==preds)

balanced_confusions = table(balanced_df) %>% as.data.frame() %>%
  mutate(
    labels = char(labels),
    preds = char(preds)
  ) %>%
  group_by(labels) %>%
  mutate(prop_gold = Freq / sum(Freq)) %>%
  ungroup() %>%
  group_by(preds) %>%
  mutate(prop_classifications = Freq / sum(Freq)) %>%
  ungroup()
balanced_diagonal = confusions %>% filter(labels==preds)
clustering = "gold"

class_freqs = unbalanced_df %>% group_by(labels) %>%
  summarise(freq=length(labels))
category_levels = c(
  "and", "but", "though", "although",
  "also", "because", "if",
  "before", "after", "while", "when", "as", "then",
  "so", "still"
)
freq_levels = class_freqs$labels[order(class_freqs$freq)]

# use *training set* frequencies
freq_levels = c('still', 'also', 'then', 'although', 'so', 'after', 'though', 'while', 'because', 'before', 'if', 'when', 'as', 'and', 'but')

freq = function(marker) {
  (class_freqs %>% subset(labels==marker))$freq[[1]]
}

unbalanced_confusions = unbalanced_confusions %>%
  mutate(labels = factor(labels, levels=freq_levels),
         preds = factor(preds, levels=freq_levels))

balanced_confusions = balanced_confusions %>%
  mutate(labels = factor(labels, levels=freq_levels),
         preds = factor(preds, levels=freq_levels))

unbalanced_confusions = unbalanced_confusions %>%
  data.frame %>%
  mutate(Confusion = log(prop_gold+0.001)) %>%
  mutate(true_freq = sapply(labels, freq)) %>%
  mutate(model_guess_freq = sapply(preds, freq))

unbalanced_p = unbalanced_confusions %>%
  mutate(Confusion = log(prop_gold+0.001)) %>%
  # mutate(Confusion = prop_gold) %>%
  ggplot(., aes(x=preds, y=labels, fill=Confusion)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_tile() +
  # facet_wrap(~corpus) +
  ylab("True Label") +
  scale_fill_gradientn(colours = c(
    "#0571b0",
    "#92c5de",
    "#f7f7f7",
    "#f4a582",
    "#ca0020")) +
  xlab("Model Classification")
print(unbalanced_p)
# ggsave("perfectly_balanced_confusion.png", width=6, height=4)
ggsave("unbalanced_confusion.png", width=6, height=4)

balanced_confusions = balanced_confusions %>%
  data.frame %>%
  mutate(Confusion = log(prop_gold+0.001)) %>%
  mutate(true_freq = sapply(labels, freq)) %>%
  mutate(model_guess_freq = sapply(preds, freq))

balanced_p = balanced_confusions %>%
  # mutate(Confusion = prop_gold) %>%
  ggplot(., aes(x=preds, y=labels, fill=Confusion)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_tile() +
  # facet_wrap(~corpus) +
  ylab("True Label") +
  scale_fill_gradientn(colours = c(
    "#0571b0",
    "#92c5de",
    "#f7f7f7",
    "#f4a582",
    "#ca0020")) +
  xlab("Model Classification")
print(balanced_p)
ggsave("balanced_confusion.png", width=6, height=4)

unbalanced_lm = unbalanced_confusions %>%
  # filter(!(labels==preds)) %>%
  lm(Confusion ~ log(model_guess_freq), .)

unbalanced_confusions = unbalanced_confusions %>%
  mutate(resid = unbalanced_lm$residuals,
         freq_predictor = unbalanced_lm$fitted.values)
balanced_confusions = balanced_confusions %>%
  mutate(resid = unbalanced_lm$residuals,
         freq_predictor = unbalanced_lm$fitted.values)

balanced_confusions %>%
  # mutate(Confusion = prop_gold) %>%
  ggplot(., aes(x=preds, y=labels, fill=freq_predictor)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_tile() +
  # facet_wrap(~corpus) +
  ylab("True Label") +
  scale_fill_gradientn(colours = c(
    "#0571b0",
    "#92c5de",
    "#f7f7f7",
    "#f4a582",
    "#ca0020")) +
  xlab("Model Classification")

balanced_confusions %>%
  # mutate(Confusion = prop_gold) %>%
  ggplot(., aes(x=preds, y=labels, fill=resid)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_tile() +
  # facet_wrap(~corpus) +
  ylab("True Label") +
  scale_fill_gradientn(colours = c(
    "#0571b0",
    "#92c5de",
    "#f7f7f7",
    "#f4a582",
    "#ca0020")) +
  xlab("Model Classification")

unbalanced_confusions %>%
  ggplot(., aes(x=log(model_guess_freq), y=Confusion)) +
  geom_point()

balanced_confusions %>%
  ggplot(., aes(x=resid, y=Confusion)) +
  geom_point() +
  xlab("Residuals of Unbalanced Confusion") +
  ylab("Balanced Confusion")
ggsave("residuals.png", width=6, height=4)

anova(unbalanced_lm)
summary(unbalanced_lm)
cor(unbalanced_confusions$Confusion, unbalanced_confusions$freq_predictor)^2
cor(balanced_confusions$Confusion, balanced_confusions$resid)^2
summary(lm(Confusion ~ resid, balanced_confusions))
anova(lm(Confusion ~ resid, balanced_confusions))



# unbalanced_confusions %>%
#   filter(!(labels==preds)) %>%
#   ggplot(., aes(x=model_guess_freq, y=Confusion)) +
#   geom_point() +
#   geom_smooth(method="loess")
# 
# # data.frame(
# #   resid = unbalanced_lm$resid,
# #   balanced_confusion = balanced_confusions$Confusion,
# #   labels = balanced_confusions$labels,
# #   preds = balanced_confusions$preds
# # ) %>%
# #   filter(preds!=labels) %>%
# #   ggplot(., aes(x=resid, y=balanced_confusion)) +
# #   geom_point() +
# #   geom_smooth(method="loess")
# 
# #   mutate(freq_ratio = model_guess_freq / true_freq) %>%
# #   # lm(accuracy ~ true_freq * model_guess_freq, .) %>%
# #   lm(accuracy ~ freq_ratio, .) %>%
# #   anova
# # 
# # unbalanced_lm = unbalanced_confusions %>%
# #   mutate(accuracy = prop_gold) %>%
# #   mutate(true_freq = sapply(labels, freq)) %>%
# #   mutate(model_guess_freq = sapply(preds, freq)) %>%
# #   select(labels, preds, model_guess_freq, true_freq, accuracy) %>%
# #   lm(accuracy ~ true_freq * model_guess_freq, .)
# # 
# # cor.test(unbalanced_lm$residuals, balanced_confusions$prop_gold)
# # 
# # data.frame(
# #   resid = unbalanced_lm$residuals,
# #   balanced_accuracy = balanced_confusions$prop_gold
# # ) %>%
# # ggplot(., aes(x=resid, y=balanced_accuracy)) +
# #   xlim(-0.1, 0.2) + 
# #   geom_point() + geom_smooth(method="loess")
# 
# unbalanced_df = unbalanced_df %>%
#   data.frame %>%
#   mutate(true_freq = sapply(labels, freq)) %>%
#   mutate(model_guess_freq = sapply(preds, freq))
# 
# balanced_df = balanced_df %>%
#   data.frame %>%
#   mutate(true_freq = sapply(labels, freq)) %>%
#   mutate(model_guess_freq = sapply(preds, freq))
# # # 

source("startup.R")
df_raw = read.csv("dissent_data.csv")
df_selected = df_raw %>%
  mutate(tag = paste(X, Epoch)) %>%
  mutate(model = X) %>%
  select(-X, Epoch, -X.1, -X.2,
         -MRPC, -STS14,
         -MRPC_ACC.1,
         -ACC_AVG) %>%
  mutate(SICKRelatedness = SICKRelatedness*100) %>%
  mutate(model = ifelse(model=="DisSent Old 5 - Tied RNN - reload, No FC", "Old DisSent 5 (arXiv, 2017)", char(model))) %>%
  mutate(model = ifelse(model=="DisSent Old 5", "DisSent but,because,if,when,so", char(model)))
df = df_selected %>% gather("Task", "Performance", 2:10)
agg = df %>% group_by(model, Epoch) %>%
  summarise(Performance = mean(Performance))
best_epochs = agg %>% group_by(model) %>%
  summarise(Epoch = Epoch[which(Performance==max(Performance))])

# # grab only best epoch, not a mix.
# # don't cheat, Erin
# df_maxes = df %>% group_by(model, Task) %>%
#   summarise(maxPerformance = max(Performance))

# filter best epochs
extract_epoch = function(m) {
  return((best_epochs %>% filter(model==m))$Epoch[[1]])
}
df_best = df %>%
  mutate(best_epoch = sapply(model, extract_epoch)) %>%
  filter(Epoch==best_epoch)

### Discourse SET
df_best %>%
  filter(model %in% c(
    "DisSent 5",
    "DisSent 8",
    "DisSent ALL"
  )) %>%
  ggplot(., aes(x=Task, y=Performance,
                colour=model, group=model,
                shape=model)) +
  geom_point() +
  geom_line() +
  theme(axis.text.x = element_text(angle=60, hjust=1)) +
  scale_shape_manual(values = 1:20)
  # scale_colour_brewer(type="qual", palette = 2)
ggsave("marker_set.png", width=6, height=4)

### Discourse SET
df_best %>%
  mutate(model = ifelse(model=="DisSent 5", "DisSent 5 (and,but,because,if,when)", char(model))) %>%
  filter(model %in% c(
    "DisSent but,because,if,when,so",
    "Old DisSent 5 (arXiv, 2017)"  
  )) %>%
  ggplot(., aes(x=Task, y=Performance,
                colour=model, group=model,
                shape=model)) +
  geom_point() +
  geom_line() +
  theme(axis.text.x = element_text(angle=60, hjust=1)) +
  scale_shape_manual(values = 1:20) +
  ggtitle("New model VS old model, same marker set")
# scale_colour_brewer(type="qual", palette = 2)
ggsave("new_vs_old_with_same_marker_set.png", width=6, height=4)

### Discourse SET
df_best %>%
  mutate(model = ifelse(model=="DisSent 5", "DisSent 5 (and,but,because,if,when)", char(model))) %>%
  filter(model %in% c(
    "DisSent 5 (and,but,because,if,when)",
    "DisSent but,because,if,when,so"
  )) %>%
  ggplot(., aes(x=Task, y=Performance,
                colour=model, group=model,
                shape=model)) +
  geom_point() +
  geom_line() +
  theme(axis.text.x = element_text(angle=60, hjust=1)) +
  scale_shape_manual(values = 1:20) +
  ggtitle("New model, new VS old marker sets")
# scale_colour_brewer(type="qual", palette = 2)
ggsave("new_model_with_new_and_old_marker_sets.png", width=6, height=4)



df_best %>%
  filter(model %in% c(
    "DisSent 5",
    "Old DisSent 5 (arXiv, 2017)"
  )) %>%
  ggplot(., aes(x=Task, y=Performance,
                colour=model, group=model,
                shape=model)) +
  geom_point() +
  geom_line() +
  theme(axis.text.x = element_text(angle=60, hjust=1)) +
  scale_shape_manual(values = 1:20)
# scale_colour_brewer(type="qual", palette = 2)
ggsave("new_vs_old.png", width=6, height=4)


df_best %>%
  filter(model %in% c(
    "DisSent 5",
    "DisSent 5 No FC"
  )) %>%
  ggplot(., aes(x=Task, y=Performance,
                colour=model, group=model,
                shape=model)) +
  geom_point() +
  geom_line() +
  theme(axis.text.x = element_text(angle=60, hjust=1)) +
  scale_shape_manual(values = 1:20)
# scale_colour_brewer(type="qual", palette = 2)
ggsave("fc_vs_no_fc.png", width=6, height=4)



df_best %>%
  filter(model %in% c(
    "DisSent 5",
    "DisSent 5 D=0.2, emb_d, FCD"
  )) %>%
  ggplot(., aes(x=Task, y=Performance,
                colour=model, group=model,
                shape=model)) +
  geom_point() +
  geom_line() +
  theme(axis.text.x = element_text(angle=60, hjust=1)) +
  scale_shape_manual(values = 1:20)
# scale_colour_brewer(type="qual", palette = 2)
ggsave("dropout_vs_no_dropout.png", width=6, height=4)


# 
# ### Ablation 5, no FC, dropout
# 
# 
# df_best %>%
#   filter(model %in% c(
#     "DisSent 5",
#     "DisSent 8",
#     "DisSent ALL"
#     # "DisSent 5 No FC",
#     # "DisSent 5 D=0.2, emb_d, FCD",
#     # "DisSent Old 5",
#     # "DisSent Old 5 - Tied RNN - reload, No FC",
#     # "DisSent 5 - Tied RNN - No reload, FC",
#     # "DisSent 5 - Tied RNN - Reload, No FC"
#   )) %>%
#   ggplot(., aes(x=Task, y=Performance,
#                 colour=model, group=model,
#                 shape=model)) +
#   geom_point(alpha=1/3) +
#   geom_line(alpha=1/3) +
#   theme(axis.text.x = element_text(angle=60, hjust=1)) +
#   scale_shape_manual(values = 1:20)
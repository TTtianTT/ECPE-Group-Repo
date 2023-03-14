source("startup.R")
df_raw = read.csv("dissent_data.csv")
df_selected = df_raw %>%
  mutate(tag = paste(X, Epoch)) %>%
  mutate(model = X) %>%
  select(-X, Epoch, -X.1, -X.2,
         -MRPC, -STS14,
         -MRPC_ACC.1, -SICKRelatedness)
df = df_selected %>% gather("Task", "Performance", 2:10)

### plot best model across epochs
#best_model = df %>% filter(model == "DisSent 5")
df %>% filter(Task!="ACC_AVG") %>%
  filter(model %in% c(
    "DisSent 5"
  )) %>%
  group_by(Epoch, model) %>%
  summarise(Performance = mean(Performance)) %>%
  ggplot(., aes(x=Epoch, y=Performance)) +
  geom_point() +
  geom_line() +
  ggtitle("DisSent 5")

### plot main models across epochs
#best_model = df %>% filter(model == "DisSent 5")
df %>% filter(Task!="ACC_AVG") %>%
  filter(model %in% c(
    "DisSent 5", "DisSent 8", "DisSent ALL"
  )) %>%
  filter(Task %in% c(
    "MR", "CR", "SUBJ", "MPQA", "SST",
    "TREC", "SICKEntailment", "MRPC_ACC", "DIS"
  )) %>%
  group_by(Epoch, model) %>%
  summarise(Performance = mean(Performance)) %>%
  ggplot(., aes(x=Epoch, y=Performance, colour=model, linetype=model)) +
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks=1:12) +
  xlab("Training Epoch") +
  ylab("Average Accuracy Across Tasks") +
  scale_colour_brewer(type="qual", palette = 6)
ggsave("epochs.png", width=8, height=5)

### plot models across epochs
best_model = df %>% filter(model == "DisSent 5")
df %>% filter(Task!="ACC_AVG") %>%
  group_by(model, Epoch) %>%
  summarise(Performance = mean(Performance)) %>%
  ggplot(., aes(x=Epoch, y=Performance, colour=model)) +
  geom_point() +
  geom_line()

df %>% filter(Task!="ACC_AVG") %>%
  filter(model %in% c(
    "DisSent 5",
    "DisSent 8",
    "DisSent ALL",
    "DisSent 5 No FC",
    "DisSent 5 D=0.2, emb_d, FCD",
    "DisSent Old 5",
    "DisSent Old 5 - Tied RNN - reload, No FC"
    #"DisSent 5 - Tied RNN - No reload, FC",   
    #"DisSent 5 - Tied RNN - Reload, No FC" 
  )) %>%
  ggplot(., aes(x=Epoch, y=Performance, colour=model)) +
  geom_line() +
  facet_wrap(~Task, scale="free")

### plot main models across epochs
#best_model = df %>% filter(model == "DisSent 5")
df %>% filter(Task!="ACC_AVG") %>%
  filter(model %in% c(
    "DisSent 5",
    "DisSent 5 No FC"
  )) %>%
  group_by(Epoch, model) %>%
  summarise(Performance = mean(Performance)) %>%
  ggplot(., aes(x=Epoch, y=Performance, colour=model)) +
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks=1:12)
ggsave("fc_vs_no_fc_epochs.png", width=6, height=4)

### plot main models across epochs
#best_model = df %>% filter(model == "DisSent 5")
df %>% filter(Task!="ACC_AVG") %>%
  filter(model %in% c(
    "DisSent 5",
    "DisSent 5 D=0.2, emb_d, FCD"
  )) %>%
  group_by(Epoch, model) %>%
  summarise(Performance = mean(Performance)) %>%
  ggplot(., aes(x=Epoch, y=Performance, colour=model)) +
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks=1:12)
ggsave("dropout_vs_no_dropout_epoch.png", width=6, height=4)

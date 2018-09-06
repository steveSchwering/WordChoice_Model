setwd("~/Documents/Psychology/Labs/LCNL/Research/Current/WordChoice_Model/ExploringModels/PredAnalysis")
d_ambig_raw = read.csv("Singleton_Ambiguous_Predictions_AllModels.csv")
d_ambig = d_ambig_raw
d_unambig_raw = read.csv("Singleton_Unambiguous_Predictions_AllModels.csv")
d_unambig = d_unambig_raw
d_ambig_sm_raw = read.csv("Singleton_Ambiguous_SustainedM_Predictions_AllModels.csv")
d_ambig_sm = d_ambig_sm_raw
d_ambig_sm_long_raw = read.csv("Singleton_Ambiguous_SustainedM_Long_Predictions_AllModels.csv")
d_ambig_sm_long = d_ambig_sm_long_raw

# Trimming
# Remove trials in which there is a matching message
d_ambig = d_ambig[d_ambig$matching_message == 0,]
d_ambig = d_ambig[!(d_ambig$matching_phonology0 == 0 & d_ambig$matching_phonology1 == 1),]
d_ambig = d_ambig[!(d_ambig$matching_phonology0 == 1 & d_ambig$matching_phonology1 == 0),]
d_ambig$critical_factor = ifelse(d_ambig$matching_phonology0 == 1,1,0) + ifelse(d_ambig$matching_phonology1==1,2,0) + ifelse(d_ambig$matching_phonology2 ==1, 4, 0)
d_ambig$critical_factor = as.factor(d_ambig$critical_factor)
levels(d_ambig$critical_factor) = c("No matching phonology", "Interference", "Priming")
d_unambig = d_unambig[d_unambig$matching_message == 0,]
d_unambig = d_unambig[!(d_unambig$matching_phonology0 == 0 & d_unambig$matching_phonology1 == 1),]
d_unambig = d_unambig[!(d_unambig$matching_phonology0 == 1 & d_unambig$matching_phonology1 == 0),]
d_unambig$critical_factor = ifelse(d_unambig$matching_phonology0 == 1,1,0) + ifelse(d_unambig$matching_phonology1==1,2,0) + ifelse(d_unambig$matching_phonology2 ==1, 4, 0)
d_unambig$critical_factor = as.factor(d_unambig$critical_factor)
levels(d_unambig$critical_factor) = c("No matching phonology", "Interference", "Priming")
d_ambig_sm = d_ambig_sm[d_ambig_sm$matching_message == 0,]
d_ambig_sm = d_ambig_sm[!(d_ambig_sm$matching_phonology0 == 0 & d_ambig_sm$matching_phonology1 == 1),]
d_ambig_sm = d_ambig_sm[!(d_ambig_sm$matching_phonology0 == 1 & d_ambig_sm$matching_phonology1 == 0),]
d_ambig_sm$critical_factor = ifelse(d_ambig_sm$matching_phonology0 == 1,1,0) + ifelse(d_ambig_sm$matching_phonology1==1,2,0) + ifelse(d_ambig_sm$matching_phonology2 ==1, 4, 0)
d_ambig_sm$critical_factor = as.factor(d_ambig_sm$critical_factor)
levels(d_ambig_sm$critical_factor) = c("No matching phonology", "Interference", "Priming")
d_ambig_sm_long = d_ambig_sm_long[d_ambig_sm_long$matching_message == 0,]
d_ambig_sm_long = d_ambig_sm_long[!(d_ambig_sm_long$matching_phonology0 == 0 & d_ambig_sm_long$matching_phonology1 == 1),]
d_ambig_sm_long = d_ambig_sm_long[!(d_ambig_sm_long$matching_phonology0 == 1 & d_ambig_sm_long$matching_phonology1 == 0),]
d_ambig_sm_long$critical_factor = ifelse(d_ambig_sm_long$matching_phonology0 == 1,1,0) + ifelse(d_ambig_sm_long$matching_phonology1==1,2,0) + ifelse(d_ambig_sm_long$matching_phonology2 ==1, 4, 0)
d_ambig_sm_long$critical_factor = as.factor(d_ambig_sm_long$critical_factor)
levels(d_ambig_sm_long$critical_factor) = c("No matching phonology", "Interference", "Priming")


# Organizing
d_ambig$modelName = as.factor(d_ambig$modelName)
#d_ambig$time = as.factor(d_ambig$time)
#levels(d_ambig$time) = c("Word 1.1", "Word 1.2", "Word 1.3", "Word 2.1", "Word 2.2", "Wordd 2.3")
d_ambig$num_batches = as.factor(d_ambig$num_batches)
levels(d_ambig$num_batches) = c("Batch size 1", "Batch size 3", "Batch size 5", "Batch size 10")
d_unambig$modelName = as.factor(d_unambig$modelName)
#d_unambig$time = as.factor(d_unambig$time)
#levels(d_unambig$time) = c("Word 1.1", "Word 1.2", "Word 1.3", "Word 2.1", "Word 2.2", "Wordd 2.3")
d_unambig$num_batches = as.factor(d_unambig$num_batches)
levels(d_unambig$num_batches) = c("Batch size 1", "Batch size 3", "Batch size 5", "Batch size 10")
d_ambig_sm$modelName = as.factor(d_ambig_sm$modelName)
#d_ambig_sm$time = as.factor(d_ambig_sm$time)
#levels(d_ambig_sm$time) = c("Word 1.1", "Word 1.2", "Word 1.3", "Word 2.1", "Word 2.2", "Wordd 2.3")
d_ambig_sm$num_batches = as.factor(d_ambig_sm$num_batches)
levels(d_ambig_sm$num_batches) = c("Batch size 1", "Batch size 3", "Batch size 5", "Batch size 10")
d_ambig_sm_long$modelName = as.factor(d_ambig_sm_long$modelName)
#d_ambig_sm_long$time = as.factor(d_ambig_sm_long$time)
#levels(d_ambig_sm_long$time) = c("Word 1.1", "Word 1.2", "Word 1.3", "Word 2.1", "Word 2.2", "Wordd 2.3")
d_ambig_sm_long$num_batches = as.factor(d_ambig_sm_long$num_batches)
levels(d_ambig_sm_long$num_batches) = c("Batch size 5", "Batch size 10")
unique(d_ambig_sm_long$num_batches)

# Graphs
#### Predictions ####
# Plotting the predictions of the model as a function of the 'time course' of the model
library("cowplot")
plotFrame = d_ambig[d_ambig$num_timeSteps == 1,]
plot_ambig_acc1 = ggplot(plotFrame, aes(x = time, y = target_value, color = num_batches)) + 
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 1.0), xlim = (c(0, 1, 2, 3, 4, 5))) + labs(title = "Model predictions", colour = "Number of batches") + ylab("Prediction") + xlab("Time step") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
plot_ambig_acc1

plotFrame = d_ambig[d_ambig$num_timeSteps == 3,]
plot_ambig_acc3 = ggplot(plotFrame, aes(x = time, y = target_value, color = num_batches)) + 
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 1.0), xlim = (c(0, 1, 2, 3, 4, 5))) + labs(title = "Model predictions", colour = "Number of batches") + ylab("Prediction") + xlab("Time step") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
plot_ambig_acc3

plotFrame = d_ambig[d_ambig$num_timeSteps == 6,]
plot_ambig_acc6 = ggplot(plotFrame, aes(x = time, y = target_value, color = num_batches)) + 
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 1.0), xlim = (c(0, 1, 2, 3, 4, 5))) + labs(title = "Model predictions", colour = "Number of batches") + ylab("Prediction") + xlab("Time step") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
plot_ambig_acc6

plotFrame = d_ambig[d_ambig$num_timeSteps == 10,]
plot_ambig_acc10 = ggplot(plotFrame, aes(x = time, y = target_value, color = num_batches)) + 
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 1.0), xlim = (c(0, 1, 2, 3, 4, 5))) + labs(title = "Model predictions", colour = "Number of batches") + ylab("Prediction") + xlab("Time step") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
plot_ambig_acc10

#### Cosine sim ####
# I should see roughly the same thing with cosine similarity
plotFrame = d_ambig[d_ambig$num_timeSteps == 3,]
plot_ambig_cos3 = ggplot(plotFrame, aes(x = time, y = cosine_sim, color = num_batches)) + 
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 1.0), xlim = (c(0, 1, 2, 3, 4, 5))) + labs(title = "Model predictions", colour = "Number of batches") + ylab("Prediction") + xlab("Time step") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
plot_ambig_cos3

plotFrame = d_ambig[d_ambig$num_timeSteps == 6,]
plot_ambig_cos6 = ggplot(plotFrame, aes(x = time, y = cosine_sim, color = num_batches)) + 
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 1.0), xlim = (c(0, 1, 2, 3, 4, 5))) + labs(title = "Model predictions", colour = "Number of batches") + ylab("Prediction") + xlab("Time step") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
plot_ambig_cos6

plotFrame = d_ambig[d_ambig$num_timeSteps == 10,]
plot_ambig_cos10 = ggplot(plotFrame, aes(x = time, y = cosine_sim, color = num_batches)) + 
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 1.0), xlim = (c(0, 1, 2, 3, 4, 5))) + labs(title = "Model predictions", colour = "Number of batches") + ylab("Prediction") + xlab("Time step") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
plot_ambig_cos10


#### Splitting up interference ####
d_ambig_word2Ambig = d_ambig[d_ambig$word2_type == 'ambiguous',]
d_ambig_word2Ambig = d_ambig_word2Ambig[d_ambig_word2Ambig$word1_type == 'unambiguous',]
d_ambig_word2Ambig = d_ambig_word2Ambig[d_ambig_word2Ambig$num_timeSteps > 5,]
d_ambig_word2Ambig$num_timeSteps = as.factor(d_ambig_word2Ambig$num_timeSteps)
d_ambig_word2Ambig = d_ambig_word2Ambig[d_ambig_word2Ambig$num_batches == 'Batch size 5',]
plot_ambig = ggplot(d_ambig_word2Ambig, aes(x = time, y = target_value, color = num_timeSteps)) + 
  geom_point(alpha = 0.5) +
  geom_jitter() +
  facet_grid(cols = vars(critical_factor)) +
  coord_cartesian(ylim = c(0.0, 1.0), xlim = c(0, 1, 2, 3, 4, 5)) + labs(title = "Model predictions", colour = "Time steps") + ylab("Prediction") + xlab("Model step") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
plot_ambig

d_ambig_word2Ambig_word2 = d_ambig_word2Ambig[d_ambig_word2Ambig$time > 2,]
varDescribeBy(d_ambig_word2Ambig_word2$target_value, d_ambig_word2Ambig_word2$critical_factor)

#### Unambiguous glance ####
d_unambig$num_batches = as.factor(d_unambig$num_batches)
plotFrame = d_unambig[d_unambig$num_timeSteps == 9,]
plot_unambig = ggplot(plotFrame, aes(x = time, y = target_value, color = num_batches)) + 
  geom_point(alpha = 0.5) +
  geom_jitter() +
  facet_grid(cols = vars(critical_factor)) +
  coord_cartesian(ylim = c(0.0, 1.0), xlim = c(0, 1, 2, 3, 4, 5)) + labs(title = "Model predictions", colour = "Number of batches") + ylab("Prediction") + xlab("Model step") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
plot_unambig

#### Ambiguous Sustained ####
# Split up into different frames
d_ambig_sm_first = d_ambig_sm[d_ambig_sm$time < 3,]
d_ambig_sm_second = d_ambig_sm[d_ambig_sm$time > 2,]

# Time step 6 -- first word
plotFrame = d_ambig_sm_first[d_ambig_sm_first$num_timeSteps == 6,]
plot_ambig_sm_first_6 = ggplot(plotFrame, aes(x = time, y = target_value, color = word1_type)) + 
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 1.0), xlim = (c(0, 1, 2, 3, 4, 5))) + labs(title = "Model predictions", colour = "Number of batches") + ylab("Prediction") + xlab("Time step") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
quartz()
plot_ambig_sm_first_6

# Time step 6 -- second word
plotFrame = d_ambig_sm_second[d_ambig_sm_second$num_timeSteps == 6,]
plot_ambig_sm_second_6 = ggplot(plotFrame, aes(x = time, y = target_value, color = word1_type)) + 
  geom_point(alpha = 0.5) +
  geom_jitter() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 1.0), xlim = (c(0, 1, 2, 3, 4, 5))) + labs(title = "Model predictions", colour = "Number of batches") + ylab("Prediction") + xlab("Time step") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
quartz()
plot_ambig_sm_second_6

# Look at all
d_ambig_sm_word2Ambig = d_ambig_sm[d_ambig_sm$word2_type == 'ambiguous',]
d_ambig_sm_word2Ambig = d_ambig_sm_word2Ambig[d_ambig_sm_word2Ambig$word1_type == 'unambiguous',]
d_ambig_sm_word2Ambig = d_ambig_sm_word2Ambig[d_ambig_sm_word2Ambig$num_timeSteps > 5,]
d_ambig_sm_word2Ambig$num_timeSteps = as.factor(d_ambig_sm_word2Ambig$num_timeSteps)
d_ambig_sm_word2Ambig = d_ambig_sm_word2Ambig[d_ambig_sm_word2Ambig$num_batches == 'Batch size 5',]
plot_ambig_sm = ggplot(d_ambig_sm_word2Ambig, aes(x = time, y = target_value, color = num_timeSteps)) + 
  geom_point(alpha = 0.5) +
  geom_jitter() +
  facet_grid(cols = vars(critical_factor)) +
  coord_cartesian(ylim = c(0.0, 1.0), xlim = c(0, 1, 2, 3, 4, 5)) + labs(title = "Model predictions", colour = "Time steps") + ylab("Prediction") + xlab("Model step") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
plot_ambig_sm
d_ambig_sm_word2Ambig_word2 = d_ambig_sm_word2Ambig[d_ambig_sm_word2Ambig$time > 2,]
varDescribeBy(d_ambig_sm_word2Ambig_word2$target_value, d_ambig_sm_word2Ambig_word2$critical_factor)

#### Ambiguous Sustained Long ####
# Split up into different frames
d_ambig_sm_long_first = d_ambig_sm_long[d_ambig_sm_long$time < 3,]
d_ambig_sm_long_second = d_ambig_sm_long[d_ambig_sm_long$time > 2,]

# Time step 6 -- first word
plotFrame = d_ambig_sm_long_first[d_ambig_sm_long_first$num_timeSteps == 6,]
plot_ambig_sm_first_6 = ggplot(plotFrame, aes(x = time, y = target_value, color = word1_type)) + 
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 1.0), xlim = (c(0, 1, 2, 3, 4, 5))) + labs(title = "Model predictions", colour = "Number of batches") + ylab("Prediction") + xlab("Time step") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
quartz()
plot_ambig_sm_first_6

# Time step 6 -- second word
plotFrame = d_ambig_sm_long_second[d_ambig_sm_long_second$num_timeSteps == 6,]
plot_ambig_sm_second_6 = ggplot(plotFrame, aes(x = time, y = target_value, color = word1_type)) + 
  geom_point(alpha = 0.5) +
  geom_jitter() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 1.0), xlim = (c(0, 1, 2, 3, 4, 5))) + labs(title = "Model predictions", colour = "Number of batches") + ylab("Prediction") + xlab("Time step") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
quartz()
plot_ambig_sm_second_6

# Look at all
d_ambig_sm_long_word2Ambig = d_ambig_sm_long[d_ambig_sm_long$word2_type == 'ambiguous',]
d_ambig_sm_long_word2Ambig = d_ambig_sm_long_word2Ambig[d_ambig_sm_long_word2Ambig$word1_type == 'unambiguous',]
d_ambig_sm_long_word2Ambig = d_ambig_sm_long_word2Ambig[d_ambig_sm_long_word2Ambig$num_timeSteps > 5,]
d_ambig_sm_long_word2Ambig$num_timeSteps = as.factor(d_ambig_sm_long_word2Ambig$num_timeSteps)
d_ambig_sm_long_word2Ambig = d_ambig_sm_long_word2Ambig[d_ambig_sm_long_word2Ambig$num_batches == 'Batch size 5',]
plot_ambig_sm_long = ggplot(d_ambig_sm_long_word2Ambig, aes(x = time, y = target_value, color = num_timeSteps)) + 
  geom_point(alpha = 0.5) +
  geom_jitter() +
  facet_grid(cols = vars(critical_factor)) +
  coord_cartesian(ylim = c(0.0, 1.0), xlim = c(0, 1, 2, 3, 4, 5)) + labs(title = "Model predictions", colour = "Time steps") + ylab("Prediction") + xlab("Model step") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
plot_ambig_sm_long
d_ambig_sm_long_word2Ambig_word2 = d_ambig_sm_long_word2Ambig[d_ambig_sm_long_word2Ambig$time > 2,]
varDescribeBy(d_ambig_sm_long_word2Ambig_word2$target_value, d_ambig_sm_long_word2Ambig_word2$critical_factor)
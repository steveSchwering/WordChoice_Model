setwd("~/Documents/Psychology/Labs/LCNL/Research/Current/WordChoice_Model/ExploringModels/LossAnalysis")
# Reading in data frames
d_ambig_loss = read.csv("Singleton_Ambiguous_AllModels.csv")
d_unambig_loss = read.csv("Singleton_Unambiguous_AllModels.csv")

# Cleaning data frames
d_ambig_loss$modelName = as.factor(d_ambig_loss$modelName)
d_unambig_loss$modelName = as.factor(d_unambig_loss$modelName)
d_ambig_loss$num_timeSteps = as.factor(d_ambig_loss$num_timeSteps)
d_unambig_loss$num_timeSteps = as.factor(d_unambig_loss$num_timeSteps)
d_ambig_loss$num_batches = as.factor(d_ambig_loss$num_batches)
levels(d_ambig_loss$num_batches) = c("Batch size 1", "Batch size 3", "Batch size 5", "Batch size 10")
d_unambig_loss$num_batches = as.factor(d_unambig_loss$num_batches)
levels(d_unambig_loss$num_batches) = c("Batch size 1", "Batch size 3", "Batch size 5", "Batch size 10")

# Creating plots
library("cowplot")
### AMBIGUOUS ###
# Looking at time steps of size 1 for ambiguous model
plotFrame = d_ambig_loss[d_ambig_loss$num_timeSteps == 1,]
plot_ambig_time1 = ggplot(plotFrame, aes(x = epoch, y = loss, color = num_batches)) + 
                          geom_point() +
                          facet_wrap(~num_batches) +
                          coord_cartesian(ylim = c(0.0, 0.033)) + labs(title = "Model losses")
plot_ambig_time1

# Looking at time steps of size 2 for ambiguous model
plotFrame = d_ambig_loss[d_ambig_loss$num_timeSteps == 2,]
plot_ambig_time2 = ggplot(plotFrame, aes(x = epoch, y = loss, color = num_batches)) + 
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 0.033)) + labs(title = "Model losses")
plot_ambig_time2

# Looking at time steps of size 3 for ambiguous model
plotFrame = d_ambig_loss[d_ambig_loss$num_timeSteps == 3,]
plot_ambig_time3 = ggplot(plotFrame, aes(x = epoch, y = loss, color = num_batches)) + 
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 0.033)) + labs(title = "Model losses")
plot_ambig_time3

# Looking at time steps of size 4 for ambiguous model
plotFrame = d_ambig_loss[d_ambig_loss$num_timeSteps == 4,]
plot_ambig_time4 = ggplot(plotFrame, aes(x = epoch, y = loss, color = num_batches)) + 
  geom_point() + 
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 0.033)) + labs(title = "Model losses")
plot_ambig_time4

# Looking at time steps of size 5 for ambiguous model
plotFrame = d_ambig_loss[d_ambig_loss$num_timeSteps == 5,]
plot_ambig_time5 = ggplot(plotFrame, aes(x = epoch, y = loss, color = num_batches)) + 
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 0.033)) + labs(title = "Model losses")
plot_ambig_time5

# Looking at time steps of size 6 for ambiguous model
plotFrame = d_ambig_loss[d_ambig_loss$num_timeSteps == 6,]
plot_ambig_time6 = ggplot(plotFrame, aes(x = epoch, y = loss, color = num_batches)) + 
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 0.033)) + labs(title = "Model losses")
plot_ambig_time6

# Looking at time steps of size 7 for ambiguous model
plotFrame = d_ambig_loss[d_ambig_loss$num_timeSteps == 10,]
plot_ambig_time10 = ggplot(plotFrame, aes(x = epoch, y = loss, color = num_batches)) + 
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 0.033)) + labs(title = "Model losses")
plot_ambig_time10

# Looking at all time steps for ambiguous model
plot_ambig = ggplot(d_ambig_loss, aes(x = epoch, y = loss, color = num_timeSteps)) +
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 0.033)) + labs(title = "Model losses", colour = "Time steps") + ylab("Loss") + xlab("Epoch") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
plot_ambig


### UNAMBIGUOUS ###
# Looking at all time steps for unambiguous model
plot_unambig = ggplot(d_unambig_loss, aes(x = epoch, y = loss, color = num_timeSteps)) +
  geom_point() +
  facet_wrap(~num_batches) +
  coord_cartesian(ylim = c(0.0, 0.033)) + labs(title = "Model losses", colour = "Time steps") + ylab("Loss") + xlab("Epoch") +
  theme(strip.text.x = element_text(size=14),
        strip.text.y = element_text(size=20, face="bold"),
        strip.background = element_rect(colour="black", fill="white"))
plot_unambig

### SAVING ###
ggsave(filename="LossAmbig_Plot.pdf", plot = plot_ambig)
ggsave(filename="LossUnambig_Plot.pdf", plot = plot_unambig)

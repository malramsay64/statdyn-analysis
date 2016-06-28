library('ggplot2')

files <- dir(pattern='*-dyn.dat')
histograms <- dir(pattern='1.40-corr.dat')

get_temp <- function(filename) {
    strsplit(filename, '-')[[1]][[3]]
}

collated <- data.frame()
for (file in files) {
    data <- read.table(file, header=TRUE)
    data["temp"] <- rep(get_temp(file), length(data$time))
    if (nrow(collated) == 0) {
        collated <- data
    }
    else {
        collated <- rbind(collated, data)
    }
}

hist <- data.frame()
for (file in histograms) {
    data <- read.table(pipe(
                    paste("awk '$1 > 50000 && $1 < 5000000 {print $0}' ",file)), 
                    header=FALSE, nrows=5e6)
    names(data) <- c("time", "coupling")
    data["temp"] <- rep(get_temp(file), length(data$time))
    data <- subset(data, data$time > 10000 && data$time < 5e6)
    if (nrow(hist) == 0) {
        hist <- data
    }
    else {
        hist <- rbind(hist, data)
    }
    break
}

collated["decoupling"] <- collated$decoupling * 0.05 * 0.05
collated <- collated[order(collated$time),]

collated_av <- aggregate(collated[,!(names(collated) %in% c("temp", "time"))],
                         list(temp=collated$temp, time=collated$time), mean)

p <- ggplot(collated_av, aes(x=time, colour=temp)) + scale_x_log10()

msd <- p + geom_path(aes(y=msd))
msd <- msd + scale_y_log10()

alpha <- p + geom_path(aes(y=alpha))

decoupling <- p + geom_path(aes(y=decoupling))

gamma1 <- p + geom_path(aes(y=gamma1))

gamma2 <- p + geom_path(aes(y=gamma2))

mean_rot <- p + geom_path(aes(y=mean_rot))
mean_rot <- mean_rot + geom_line(aes(y=pi/2), colour="grey")

correlations <- p + geom_path(aes(y=correlation))

corr_hist <- ggplot(hist) 
corr_hist <- corr_hist + geom_density(aes(x=coupling, colour=as.factor(time))) 
corr_hist <- corr_hist + scale_x_continuous(limits=c(-20,20))
corr_hist <- corr_hist + scale_y_continuous(limits=c(0,1))
corr_hist <- corr_hist + labs(title=hist$temp[[1]])

p_sm <- ggplot(collated, aes(x=time, colour=temp))
msd_sm <- p_sm + geom_smooth(aes(y=msd))
msd_sm <- msd_sm + scale_y_log10()

alpha_sm <- p_sm + geom_smooth(aes(y=alpha))

decoupling_sm <- p_sm + geom_smooth(aes(y=decoupling))

pdf("dynamics.pdf", width=8, height=6)
print(msd)
print(alpha)
print(decoupling)
print(gamma1)
print(gamma2)
print(mean_rot)
print(correlations)
print(corr_hist)
#print(msd_sm)
#print(alpha_sm)
#print(coupling_sm)
dev.off()

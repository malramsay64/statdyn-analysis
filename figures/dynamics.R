library('ggplot2')

files <- dir(pattern='*-dyn.dat')

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

rot1 <- p + geom_path(aes(y=rot1))

rot2 <- p + geom_path(aes(y=rot2))

param_rot_n3 <- p + geom_path(aes(y=param_rot_n3))
param_rot_n2 <- p + geom_path(aes(y=param_rot_n2))
param_rot_n1 <- p + geom_path(aes(y=param_rot_n1))
param_rot_n0.1 <- p + geom_path(aes(y=param_rot_n0.1))
param_rot_0.1 <- p + geom_path(aes(y=param_rot_0.1))
param_rot_1 <- p + geom_path(aes(y=param_rot_1))
param_rot_2 <- p + geom_path(aes(y=param_rot_2))
param_rot_3 <- p + geom_path(aes(y=param_rot_3))


param_trans_n3 <- p + geom_path(aes(y=param_trans_n3))
param_trans_n2 <- p + geom_path(aes(y=param_trans_n2))
param_trans_n1 <- p + geom_path(aes(y=param_trans_n1))
param_trans_n0.1 <- p + geom_path(aes(y=param_trans_n0.1))
param_trans_0.1 <- p + geom_path(aes(y=param_trans_0.1))
param_trans_1 <- p + geom_path(aes(y=param_trans_1))
param_trans_2 <- p + geom_path(aes(y=param_trans_2))
param_trans_3 <- p + geom_path(aes(y=param_trans_3))

pdf("dynamics.pdf", width=8, height=6)
print(msd)
print(alpha)
print(decoupling)
print(gamma1)
print(gamma2)
print(mean_rot)
print(rot1)
print(rot2)
print(param_rot_n3)
print(param_rot_n2)
print(param_rot_n1)
print(param_rot_n0.1)
print(param_rot_0.1)
print(param_rot_1)
print(param_rot_2)
print(param_rot_3)
print(param_trans_n3)
print(param_trans_n2)
print(param_trans_n1)
print(param_trans_n0.1)
print(param_trans_0.1)
print(param_trans_1)
print(param_trans_2)
print(param_trans_3)
dev.off()

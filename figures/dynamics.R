library('ggplot2')
library('reshape2')
library('plotly')

files <- dir(pattern='*-dyn.dat')
timestep <- 0.005

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

p <- ggplot(collated_av, aes(x=time*titimestep, colour=temp)) + scale_x_log10()

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

diff <- ggplot(subset(collated_av, collated_av$temp == "2.00"), aes(x=time))
diff_trans <- diff + geom_path(aes(y=param_trans_n3-disp, colour="-3"))
diff_trans <- diff_trans + geom_path(aes(y=param_trans_n2-disp, colour="-2"))
diff_trans <- diff_trans + geom_path(aes(y=param_trans_n1-disp, colour="-1"))
diff_trans <- diff_trans + geom_path(aes(y=param_trans_n0.1-disp, colour="-0.1"))
diff_trans <- diff_trans + geom_path(aes(y=param_trans_0.1-disp, colour="0.1"))
diff_trans <- diff_trans + geom_path(aes(y=param_trans_1-disp, colour="1"))
diff_trans <- diff_trans + geom_path(aes(y=param_trans_2-disp, colour="2"))
diff_trans <- diff_trans + geom_path(aes(y=param_trans_3-disp, colour="3"))

diff_rot <- diff + geom_path(aes(y=param_rot_n3-mean_rot, colour="-3"))
diff_rot <- diff_rot + geom_path(aes(y=param_rot_n2-mean_rot, colour="-2"))
diff_rot <- diff_rot + geom_path(aes(y=param_rot_n1-mean_rot, colour="-1"))
diff_rot <- diff_rot + geom_path(aes(y=param_rot_n0.1-mean_rot, colour="-0.1"))
diff_rot <- diff_rot + geom_path(aes(y=param_rot_0.1-mean_rot, colour="0.1"))
diff_rot <- diff_rot + geom_path(aes(y=param_rot_1-mean_rot, colour="1"))
diff_rot <- diff_rot + geom_path(aes(y=param_rot_2-mean_rot, colour="2"))
diff_rot <- diff_rot + geom_path(aes(y=param_rot_3-mean_rot, colour="3"))

trans_heading <- grep("param_trans", names(collated_av), value=TRUE)
conv_trans <- list(
    "time"="time",
    "temp"="temp",
    "param_trans_n3"=-3,
    "param_trans_n2"=-2,
    "param_trans_n1"=-1,
    "param_trans_n0.1"=-0.1,
    "param_trans_0.1"=0.1,
    "param_trans_1"=1,
    "param_trans_2"=2,
    "param_trans_3"=3
    )
trans_heading <- c(grep("param_trans", names(collated_av), value=TRUE), "time", "temp")
trans_2d <- collated_av[trans_heading]
names(trans_2d) <- conv_trans[names(trans_2d)]
trano_2d <- subset(trans_2d, trans_2d$temp == "1.40")
trans_2dm <- melt(trans_2d, id.vars = c("time", "temp"))
t2d <- plot_ly(trans_2dm, x=log(time), y=variable, z=log(value), type="scatter3d", group=temp, mode="markers")
print(t2d)

pdf("dynamics.pdf", width=8, height=6)
print(msd)
print(alpha)
print(decoupling)
print(gamma1)
print(gamma2)
print(mean_rot)
print(rot1)
print(rot2)
print(diff_trans)
print(diff_rot)
dev.off()

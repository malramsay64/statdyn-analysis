library('ggplot2')

files <- dir(pattern='*0-dyn.dat')
files_trans <- dir(pattern='*-dyn2.dat')
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

collated_trans <- data.frame()
for (file in files_trans) {
    data <- read.table(file, header=TRUE)
    names(data) <- c("time", "displacement", "MSD", "MFD",  "alpha", "struct")
    data["temp"] <- rep(get_temp(file), length(data$time))
    if (nrow(collated_trans) == 0) {
        collated_trans <- data
    }
    else {
        collated_trans <- rbind(collated_trans, data)
    }
}

collated["decoupling"] <- collated$decoupling * 0.05 * 0.05
collated <- collated[order(collated$time),]
collated_trans <- collated_trans[order(collated_trans$time),]

collated_av <- aggregate(collated[,!(names(collated) %in% c("temp", "time"))],
                         list(temp=collated$temp, time=collated$time), mean)
collated_trans <- aggregate(collated_trans[,!(names(collated_trans) %in% c("temp", "time"))],
                         list(temp=collated_trans$temp, time=collated_trans$time), mean)

collated_av <- subset(collated_av, collated_av$temp >= 1.40)

p <- ggplot(collated_av, aes(x=time*timestep, colour=temp)) + scale_x_log10() + labs(x="Time")
p <- p + theme_bw(base_size=19) + scale_color_discrete("Temperature")

msd <- p + geom_path(aes(y=msd))
msd <- msd + scale_y_log10() + labs(y=expression(symbol("\341")* Delta*r^2* symbol("\361")))

alpha <- p + geom_path(aes(y=alpha)) + labs(y=expression(alpha))
#+ labs(y=expression(frac(symbol("\341") *Delta *r^4 *symbol("\361"), 2*symbol("\341")* Delta *r^2 *symbol("\361")^2) -1))

decoupling <- p + geom_path(aes(y=decoupling))

g1 <- expression(frac(symbol("\341") *Delta *r *Delta*theta* symbol("\361") - symbol("\341")* Delta*r*symbol("\361")* symbol("\341")* Delta*theta*symbol("\361"), symbol("\341")* Delta * r *symbol("\361")* symbol("\341") *Delta * theta *symbol("\361")))
g2 <- expression(frac(symbol("\341") *(Delta *r *Delta*theta)^2* symbol("\361") - symbol("\341")* Delta*r^2*symbol("\361")* symbol("\341")* Delta*theta^2*symbol("\361"), symbol("\341")* Delta * r^2 *symbol("\361")* symbol("\341") *Delta * theta^2* symbol("\361")))
gamma1 <- p + geom_path(aes(y=gamma1)) + labs(y=expression(gamma[1]))

gamma2 <- p + geom_path(aes(y=gamma2)) + labs(y=g2)

mean_rot <- p + geom_path(aes(y=mean_rot)) + labs(y=expression(symbol("\341")*Delta*theta*symbol("\361")))
mean_rot <- mean_rot + geom_line(aes(y=pi/2), colour="grey")

rot1 <- p + geom_path(aes(y=rot1))

rot2 <- p + geom_path(aes(y=rot2))

trans_corel <- p + geom_path(aes(y=trans_corel)) 
rot_corel <- p + geom_path(aes(y=rot_corel))

diff <- ggplot(subset(collated_av, collated_av$temp == "1.40"), aes(x=time))
diff <- diff + scale_color_discrete(expression(alpha), labels=c("-3"="-3", "-2"="-2", "-1"="-1", "-0.1"="-0.1", "0.1"="0.1", "1"="1", "2"="2", "3"="3"))
diff_trans <- diff + geom_path(aes(y=param_trans_n3-disp, colour="-3"))
diff_trans <- diff_trans + geom_path(aes(y=param_trans_n2-disp, colour="-2"))
diff_trans <- diff_trans + geom_path(aes(y=param_trans_n1-disp, colour="-1"))
diff_trans <- diff_trans + geom_path(aes(y=param_trans_n0.1-disp, colour="-0.1"))
diff_trans <- diff_trans + geom_path(aes(y=param_trans_0.1-disp, colour="0.1"))
diff_trans <- diff_trans + geom_path(aes(y=param_trans_1-disp, colour="1"))
diff_trans <- diff_trans + geom_path(aes(y=param_trans_2-disp, colour="2"))
diff_trans <- diff_trans + geom_path(aes(y=param_trans_3-disp, colour="3"))

diff_rot <- diff + geom_path(aes(y=param_rot_n3-mean_rot, colour=as.factor(-3)))
diff_rot <- diff_rot + geom_path(aes(y=param_rot_n2-mean_rot, colour=as.factor(-2)))
diff_rot <- diff_rot + geom_path(aes(y=param_rot_n1-mean_rot, colour=as.factor(-1)))
diff_rot <- diff_rot + geom_path(aes(y=param_rot_n0.1-mean_rot, colour=as.factor(-0.1)))
diff_rot <- diff_rot + geom_path(aes(y=param_rot_0.1-mean_rot, colour=as.factor(0.1)))
diff_rot <- diff_rot + geom_path(aes(y=param_rot_1-mean_rot, colour=as.factor(1)))
diff_rot <- diff_rot + geom_path(aes(y=param_rot_2-mean_rot, colour=as.factor(2)))
diff_rot <- diff_rot + geom_path(aes(y=param_rot_3-mean_rot, colour=as.factor(3)))

struct_relax <- ggplot(collated_trans, aes(x=time, y=struct)) + geom_path()
struct_relax <- struct_relax + labs(x="Time", y="Structural Relaxation")


rot1_relax = list()
for (t in unique(collated_av$temp)) {
    s <- subset(collated_av, collated_av$temp == t)
    rot1_relax[t] <- s$time[[Position(function(i) i < 1/exp(1),s$rot1)]]
}
rot1_df <- data.frame(x=1/as.numeric(names(rot1_relax)), y=as.numeric(rot1_relax))
rot1_rplot <- ggplot(rot1_df, aes(x=x, y=y)) + geom_point() + geom_line() + scale_y_log10()
rot1_rplot <- rot1_rplot + labs(x="1/T", y="Rotational Relaxation (R1) Time")

rot2_relax = list()
for (t in unique(collated_av$temp)) {
    s <- subset(collated_av, collated_av$temp == t)
    rot2_relax[t] <- s$time[[Position(function(i) i < 1/exp(1),s$rot2)]]
}
rot2_df <- data.frame(x=1/as.numeric(names(rot2_relax)), y=as.numeric(rot2_relax))
rot2_rplot <- ggplot(rot2_df, aes(x=x , y=y )) + geom_point() + geom_line() + scale_y_log10()
rot2_rplot <- rot2_rplot + labs(x="1/T", y="Rotational Relaxation (R2) Time")

struct_relax = list()
for (t in unique(collated_trans$temp)) {
    s <- subset(collated_trans, collated_trans$temp == t)
    struct_relax[t] <- s$time[[Position(function(i) i < 1/exp(1),s$struct)]]
}
struct_df <- data.frame(x=1/as.numeric(names(struct_relax)), y=as.numeric(struct_relax))
struct_rplot <- ggplot(struct_df, aes(x=x , y=y )) + geom_point() + geom_line() + scale_y_log10()
struct_rplot <- struct_rplot + labs(x="1/T", y="Structural Relaxation Time")

max_g1 <- list()
max_g2 <- list()
max_alpha <- list()
for (t in unique(collated_av$temp)) {
    s <- subset(collated_av, collated_av$temp == t)
    max_g1[t] <- max(s$gamma1)
    max_g2[t] <- max(s$gamma2)
    max_alpha[t] <- max(s$alpha)
}
m_gamma_df <- data.frame(temp=as.numeric(names(max_g1)), max_g1=as.numeric(max_g1), max_g2=as.numeric(max_g2))
m_g <- ggplot(m_gamma_df, aes(x=1/temp)) + labs(x="1/T")

mg1 <- expression(max(frac(symbol("\341") *Delta *r *Delta*theta* symbol("\361") - symbol("\341")* Delta*r*symbol("\361")* symbol("\341")* Delta*theta*symbol("\361"), symbol("\341")* Delta * r *symbol("\361")* symbol("\341") *Delta * theta *symbol("\361"))))
mg2 <- expression(max(frac(symbol("\341") *(Delta *r *Delta*theta)^2* symbol("\361") - symbol("\341")* Delta*r^2*symbol("\361")* symbol("\341")* Delta*theta^2*symbol("\361"), symbol("\341")* Delta * r^2 *symbol("\361")* symbol("\341") *Delta * theta^2* symbol("\361"))))
m_g1 <- m_g + geom_line(aes(y=max_g1)) + geom_point(aes(y=max_g1)) + labs(y=mg1)
m_g2 <- m_g + geom_line(aes(y=max_g2)) + geom_point(aes(y=max_g2)) + labs(y=mg2)
m_alpha <- m_g + geom_line(aes(y=max_alpha)) + geom_point(aes(y=max_alpha)) + labs(y=expression(max(alpha))

D = list()
for (t in unique(collated_av$temp)) {
    s <- subset(collated_av, collated_av$temp == t)
    D[t] = s$time[[Position(function(i) i < 1/exp(1),s$rot2)]]
}

pdf("dynamics.pdf", width=10, height=6)
print(msd)
print(alpha)
print(decoupling)
print(gamma1)
print(gamma2)
print(mean_rot)
print(rot1)
print(rot2)
print(trans_corel)
print(rot_corel)
print(diff_trans)
print(diff_rot)
print(rot1_rplot)
print(rot2_rplot)
print(struct_rplot)
print(m_g1)
print(m_g2)
print(m_alpha)
dev.off()

pdf("gamma1.pdf", width=10, height=6)
print(gamma1)
dev.off()

pdf("alpha.pdf", width=10,height=6)
print(alpha)
dev.off()

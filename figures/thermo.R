library('ggplot2')

files <- dir(pattern='*-thermo.dat')

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

p <- ggplot(collated, aes(x=timestep, colour=temp))
p <- p + scale_x_log10() 

temp <- p + geom_smooth(aes(y=temperature), level=0.999)
press <- p + geom_smooth(aes(y=pressure), level=0.999)
energy_potential <- p + geom_smooth(aes(y=abs(potential_energy)), level=0.999)
energy_kinetic <- p + geom_smooth(aes(y=translational_kinetic_energy+rotational_kinetic_energy), level=0.999)

pdf("thermo.pdf", width=8, height=6)
print(temp)
print(press)
print(energy_potential)
print(energy_kinetic)
dev.off()

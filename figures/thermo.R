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
thermo <- p + geom_smooth(aes(y=temperature))
thermo <- thermo + geom_smooth(aes(y=pressure))

pdf("thermo.pdf", width=8, height=6)
print(thermo)
dev.off()

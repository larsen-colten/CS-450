library(datasets)
data = state.x77

distance = dist(as.matrix(data))
hc = hclust(distance)

plot(hc)

data_scaled = scale(data)
distance = dist(as.matrix(data_scaled))
hc = hclust(distance)

plot(hc)

data = subset(data, select=-c(Area))
data_scaled = scale(data)
distance = dist(as.matrix(data_scaled))
hc = hclust(distance)

plot(hc)

data = subset(data, select=c(Frost))
data_scaled = scale(data)
distance = dist(as.matrix(data_scaled))
hc = hclust(distance)

plot(hc)
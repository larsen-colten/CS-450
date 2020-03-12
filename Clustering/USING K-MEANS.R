library(datasets)
data = state.x77

data = scale(data)

table = NULL;
for (i in 1:25) {
  clusters = kmeans(data, i)
  table[i] = clusters$tot.withinss
}
plot(table)

clusters = kmeans(data, 10)
summary(clusters)
clusters$centers
#clusters$cluster
#clusters$withinss
clusters$tot.withinss

library(cluster)
clusplot(data, clusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)

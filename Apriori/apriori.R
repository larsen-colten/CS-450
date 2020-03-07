# install.packages('arules');
library(arules);
data(Groceries);

frequentItems <- eclat (Groceries, parameter = list(supp = 0.07, maxlen = 15))
inspect(frequentItems)

rules <- apriori (Groceries, parameter = list(supp = 0.001, conf = 0.5))

rules_conf <- sort (rules, by="confidence", decreasing=TRUE)
inspect(head(rules_conf))

rules_lift <- sort (rules, by="lift", decreasing=TRUE)
inspect(head(rules_lift))

rules <- apriori (data=Groceries, parameter=list (supp=0.001,conf = 0.15,minlen=2), appearance = list(default="rhs",lhs="shopping bags"), control = list (verbose=F))
rules_conf <- sort (rules, by="conf", decreasing=TRUE)
inspect(head(rules_conf))


# {soda} => {other vegetables}
# {yogurt} => {root vegetables}
# {canned fish,hygiene articles} => {whole milk}
# {sausage} => {shopping bags}
# {shopping bags} => {soda}
# install.packages('arules');
library(arules);
data(Groceries);

frequentItems <- eclat (Groceries, parameter = list(supp = 0.1, maxlen = 15))
inspect(frequentItems)

rules <- apriori (Groceries, parameter = list(supp = 0.001, conf = 0.5))

rules_supp <- sort (rules, by="support", decreasing=TRUE)
inspect(head(rules_supp))

rules_conf <- sort (rules, by="confidence", decreasing=TRUE)
inspect(head(rules_conf))

rules_lift <- sort (rules, by="lift", decreasing=TRUE)
inspect(head(rules_lift))


appearance = list(default="rhs",lhs="soda")

rules <- apriori (data=Groceries, parameter=list (supp=0.001,conf = 0.15,minlen=2), appearance = list(rhs="soda",lhs="shopping bags"))
rule <- sort (rules, by="confidence", decreasing=TRUE)
inspect(head(rule))

# Interesting rules
# {soda} => {other vegetables}
# {yogurt} => {root vegetables}
# {canned fish,hygiene articles} => {whole milk}
# {sausage} => {shopping bags}
# {shopping bags} => {soda}

itemFrequencyPlot(Groceries, topN=10, type="absolute", main="Item Frequency")
